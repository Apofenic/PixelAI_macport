-- Local AI Generator for Aseprite - Enhanced UI Version
-- Professional pixel art generation using Stable Diffusion
-- (Debug note) Removed prior intentional error to allow UI to load while we surface path info.
local script_path = debug.getinfo(1, "S").source:sub(2)
local script_dir = script_path:match("(.*)[/\\]")

local function safe_dofile(filename)
    local full_path = script_dir .. "/" .. filename
    local success, result = pcall(dofile, full_path)
    if not success then 
        app.alert("Error loading " .. filename .. ": " .. tostring(result))
        return nil
    end
    return result
end

-- Load required libraries
local json = safe_dofile("json.lua")
local base64 = safe_dofile("base64.lua") 
local http_client = safe_dofile("http-client.lua")

if not json or not base64 or not http_client then 
    app.alert("Failed to load required libraries. Please ensure all files are in the same directory.")
    return 
end

-- Plugin configuration
local plugin_config = {
    server_url = "http://127.0.0.1:5000",
    name = "Local AI Generator v2.0",
    version = "2.0.0"
}

-- DEBUG: capture script dir fingerprint so user can verify which copy is running without console
local debug_fingerprint = (script_dir or "?")
    :gsub(os.getenv("HOME") or "", "~")
local function short_path(p)
    if #p > 48 then
        return "…" .. p:sub(-47)
    end
    return p
end

-- Global state
local is_generating = false
local current_dialog = nil
local available_models = {}
local available_loras = {}
local lora_value_by_label = {}
local lora_label_by_value = {}
local server_status = "Unknown"
local last_generation_time = 0
local current_engine = "torch" -- 'torch' or 'mlx'

-- Enhanced default settings with better defaults for pixel art
local default_settings = {
    prompt = "pixel art, cute character, 16-bit style, vibrant colors",
    negative_prompt = "blurry, smooth, antialiased, realistic, photographic, 3d render, low quality",
    pixel_width = 64,
    pixel_height = 64,
    steps = 25,
    guidance_scale = 7.5,
    colors = 16,
    seed = -1,
    remove_background = false,
    model_name = "stabilityai/stable-diffusion-xl-base-1.0",
    lora_model = "nerijs/pixel-art-xl",
    lora_strength = 0.8,
    output_method = "New Layer",
    generation_quality = "High (1024x1024)"
}

-- Current settings (copy of defaults)
local current_settings = {}
for k, v in pairs(default_settings) do 
    current_settings[k] = v 
end

-- Preset prompts for quick access
local preset_prompts = {
    "pixel art, cute animal, simple design, flat colors",
    "pixel art, fantasy character, rpg style, detailed sprite",
    "pixel art, sci-fi robot, futuristic, metallic",
    "pixel art, magical item, glowing, mystical",
    "pixel art, food item, colorful, appetizing",
    "pixel art, vehicle, side view, detailed",
    "pixel art, building, isometric view, architectural",
    "pixel art, nature scene, trees, peaceful"
}

-- Quality presets for base generation resolution
local quality_presets = {
    {name = "Fast (512x512)", width = 512, height = 512, description = "Faster generation"},
    {name = "High (1024x1024)", width = 1024, height = 1024, description = "Better quality (default)"},
    {name = "Ultra (1536x1536)", width = 1536, height = 1536, description = "Best quality (slow)"},
    {name = "Max (2048x2048)", width = 2048, height = 2048, description = "Maximum quality (very slow)"}
}

-- Common dimension presets
local dimension_presets = {
    {name = "Tiny (32x32)", width = 32, height = 32},
    {name = "Small (64x64)", width = 64, height = 64},
    {name = "Medium (128x128)", width = 128, height = 128},
    {name = "Large (256x256)", width = 256, height = 256},
    {name = "Portrait (64x96)", width = 64, height = 96},
    {name = "Landscape (96x64)", width = 96, height = 64}
}

-- Utility functions
local function format_time(seconds)
    if seconds < 60 then
        return string.format("%.1fs", seconds)
    else
        return string.format("%.1fm", seconds / 60)
    end
end

local function prepare_image_for_generation(output_method, image_mode)
    if not app.activeSprite.selection.isEmpty then 
        app.command.Cancel() 
    end
    
    local cel
    app.transaction("AI Generation Setup", function()
        local timestamp = os.date("%H:%M:%S")
        local layer_name = "AI Gen " .. timestamp
        local layer = app.activeSprite:newLayer{name = layer_name, colorMode = image_mode}
        app.activeLayer = layer
        
        local frame
        if output_method == "New Frame" then
            frame = app.activeSprite:newEmptyFrame(app.activeFrame.frameNumber + 1)
        else
            frame = app.activeFrame
        end
        
        cel = app.activeSprite:newCel(layer, frame)
    end)
    return cel
end

local function fetch_models_and_loras(callback)
    server_status = "Connecting..."
    
    http_client.get(plugin_config.server_url .. "/models", function(res, err)
        if res and res.models then
            available_models = res.models
        else
            available_models = {"Connection Failed"}
        end
        
        http_client.get(plugin_config.server_url .. "/loras", function(res2, err2)
            -- rebuild lists with compatibility labels
            available_loras = {}
            lora_value_by_label = {}
            lora_label_by_value = {}
            if res2 and res2.loras then
                local compat = {}
                if res2.compatible_loras then
                    for _,v in ipairs(res2.compatible_loras) do compat[v] = true end
                end
                for _,v in ipairs(res2.loras) do
                    local label = v
                    if v ~= "None" and not compat[v] then
                        label = v .. " (incompatible)"
                    end
                    table.insert(available_loras, label)
                    lora_value_by_label[label] = v
                    lora_label_by_value[v] = label
                end
            else
                available_loras = {"Connection Failed"}
            end
            
            -- Check server health
            http_client.get(plugin_config.server_url .. "/health", function(health_res, health_err)
                if health_res then
                    current_engine = health_res.engine or current_engine
                    server_status = "Online" .. (current_engine and (" [" .. current_engine .. "]") or "")
                    if health_res.current_model then
                        server_status = server_status .. " - " .. health_res.current_model
                    end
                else
                    server_status = "Offline"
                end

                if callback then callback() end
            end)
        end)
    end)
end

local function generate_image(settings, callback)
    if is_generating then 
        app.alert("Generation already in progress. Please wait...")
        return 
    end
    
    is_generating = true
    local start_time = os.clock()
    
    -- Get base generation resolution from quality preset
    local base_width = 1024  -- default
    local base_height = 1024 -- default
    
    for _, preset in ipairs(quality_presets) do
        if preset.name == settings.generation_quality then
            base_width = preset.width
            base_height = preset.height
            break
        end
    end
    
    local request_data = {
        prompt = settings.prompt,
        negative_prompt = settings.negative_prompt,
        width = base_width,  -- Base generation resolution
        height = base_height, -- Base generation resolution
        pixel_width = settings.pixel_width,   -- Final pixel art size
        pixel_height = settings.pixel_height, -- Final pixel art size
        steps = settings.steps,
        guidance_scale = settings.guidance_scale,
        colors = settings.colors,
        lora_model = settings.lora_model,
        remove_background = settings.remove_background,
        lora_strength = settings.lora_strength
    }
    
    -- Only include seed if it's not -1 (random)
    if settings.seed ~= -1 then
        request_data.seed = settings.seed
    end
    
    http_client.post(plugin_config.server_url .. "/generate", request_data, function(response, error)
        is_generating = false
        last_generation_time = os.clock() - start_time
        
        if callback then 
            callback(response, error) 
        end
    end)
end

local function place_image_in_aseprite_raw(image_data, output_method)
    local image_mode = (image_data.mode == "rgba") and ColorMode.RGBA or ColorMode.RGB
    
    -- Create new sprite if none exists
    if not app.activeSprite then
        app.command.NewFile{
            width = image_data.width,
            height = image_data.height,
            colorMode = image_mode
        }
    end
    
    local cel = prepare_image_for_generation(output_method, image_mode)
    if not cel then 
        app.alert("Could not prepare canvas for image placement.")
        return 
    end
    
    local pixel_data = base64.decode(image_data.base64)
    
    app.transaction("Place AI Generated Image", function()
        local im = Image(image_data.width, image_data.height, image_mode)
        
        -- Safely set image bytes
        local success = pcall(function() 
            im.bytes = pixel_data 
        end)
        
        if success then
            cel.image:clear()
            cel.image:drawImage(im, Point(0, 0))
        else
            app.alert("Failed to process image data.")
        end
    end)
    
    app.refresh()
end

local function update_dialog_status(dlg)
    if dlg and dlg.data then
        -- Update server status
        dlg:modify{id="server_status_label", text="Status: " .. server_status}
        
        -- Update generation button
        if is_generating then
            dlg:modify{
                id = "generate", 
                enabled = false, 
                text = "Generating..."
            }
        else
            dlg:modify{
                id = "generate", 
                enabled = true, 
                text = "Generate Image"
            }
        end
        
        -- Update last generation time
        if last_generation_time > 0 then
            dlg:modify{
                id = "generation_time_label",
                text = "Last generation: " .. format_time(last_generation_time)
            }
        end
    end
end

-- Dialog state tracking
local show_advanced = false
local show_model_settings = false

local function open_advanced_dialog()
    local adv_dlg = Dialog("Advanced Settings")
    
    adv_dlg:slider{
        id = "steps",
        label = "Quality Steps:",
        min = 10,
        max = 50,
        value = current_settings.steps,
        onchange = function()
            current_settings.steps = adv_dlg.data.steps
        end
    }
    
    adv_dlg:slider{
        id = "guidance_scale",
        label = "Prompt Adherence:",
        min = 1,
        max = 20,
        value = current_settings.guidance_scale,
        onchange = function()
            current_settings.guidance_scale = adv_dlg.data.guidance_scale
        end
    }
    
    adv_dlg:number{
        id = "seed",
        label = "Seed (-1 = random):",
        text = tostring(current_settings.seed),
        decimals = 0,
        onchange = function()
            current_settings.seed = adv_dlg.data.seed
        end
    }
    
    adv_dlg:entry{
        id = "negative_prompt",
        label = "Negative Prompt:",
        text = current_settings.negative_prompt,
        onchange = function()
            current_settings.negative_prompt = adv_dlg.data.negative_prompt
        end
    }
    
    adv_dlg:button{text = "Close", onclick = function() adv_dlg:close() end}
    adv_dlg:show{wait = false}
end

local function open_model_dialog()
    local model_dlg = Dialog("Model Settings")

    model_dlg:combobox{
        id = "engine",
        label = "Engine:",
        options = {"torch", "mlx"},
        option = current_engine,
        onchange = function()
            local new_engine = model_dlg.data.engine
            if new_engine ~= current_engine then
                http_client.post(plugin_config.server_url .. "/set_engine", { engine = new_engine }, function(res, err)
                    if res and res.success then
                        current_engine = new_engine
                        if current_engine == "mlx" then
                            app.alert("MLX engine selected. Generation will use a placeholder until real MLX support is implemented.")
                        end
                        fetch_models_and_loras(function()
                            -- Refresh model & lora controls enable state
                            local enable_models = (current_engine == "torch")
                            model_dlg:modify{ id = "model_name", enabled = enable_models }
                            model_dlg:modify{ id = "lora_model", enabled = enable_models }
                        end)
                    else
                        app.alert("Failed to switch engine: " .. tostring(err or (res and res.error) or "Unknown"))
                        model_dlg:modify{ id = "engine", option = current_engine }
                    end
                end)
            end
        end
    }
    
    model_dlg:combobox{
        id = "model_name",
        label = "Base Model:",
        options = available_models,
        option = current_settings.model_name,
        onchange = function()
            current_settings.model_name = model_dlg.data.model_name
            http_client.post(plugin_config.server_url .. "/load_model", { model_name = current_settings.model_name }, function()
                fetch_models_and_loras(function()
                    if model_dlg then
                        local new_label = lora_label_by_value[current_settings.lora_model] or current_settings.lora_model
                        model_dlg:modify{ id = "lora_model", options = available_loras, option = new_label }
                    end
                end)
            end)
        end
    }
    
    model_dlg:separator{text = "Generation Quality"}
    
    model_dlg:combobox{
        id = "generation_quality",
        label = "Base Resolution:",
        options = (function()
            local options = {}
            for _, preset in ipairs(quality_presets) do
                table.insert(options, preset.name)
            end
            return options
        end)(),
        option = current_settings.generation_quality,
        onchange = function()
            current_settings.generation_quality = model_dlg.data.generation_quality
        end
    }
    
    model_dlg:label{text = "Higher resolution = better pixel art quality but slower generation"}
    
    model_dlg:separator{text = "Art Style"}
    
    local current_lora_label = lora_label_by_value[current_settings.lora_model] or current_settings.lora_model
    model_dlg:combobox{
        id = "lora_model",
        label = "Art Style (LoRA):",
        options = available_loras,
        option = current_lora_label,
        onchange = function()
            local chosen_label = model_dlg.data.lora_model
            local actual_value = lora_value_by_label[chosen_label] or chosen_label
            current_settings.lora_model = actual_value
            if chosen_label:find('%(incompatible%)') then
                app.alert("Selected LoRA may be incompatible and could fail to generate.")
            end
        end
    }
    
    model_dlg:slider{
        id = "lora_strength",
        label = "Style Strength:",
        min = 0,
        max = 2,
        value = current_settings.lora_strength,
        onchange = function()
            current_settings.lora_strength = model_dlg.data.lora_strength
        end
    }
    
    model_dlg:button{text = "Close", onclick = function() model_dlg:close() end}
    model_dlg:show{wait = false}

    -- Disable model & LoRA controls if MLX (placeholder) engine
    if current_engine == "mlx" then
        model_dlg:modify{ id = "model_name", enabled = false }
        model_dlg:modify{ id = "lora_model", enabled = false }
    end
end

local function create_main_dialog()
    if current_dialog then 
        current_dialog:close() 
    end
    
    local dlg = Dialog("Local AI Generator blah")
    current_dialog = dlg
    
    -- Server status + debug path
    dlg:label{id="server_status_label", text="Status: " .. server_status}
    dlg:label{id="debug_path_label", text="Path: " .. short_path(debug_fingerprint)}
    dlg:button{text="Copy Path", onclick=function()
        -- Aseprite has no native clipboard API in all versions; show via alert for copy
        app.alert("Current script directory:\n" .. debug_fingerprint)
    end}
    dlg:button{
        text="Refresh Connection", 
        onclick=function()
            dlg:modify{id="server_status_label", text="Status: Checking..."}
            fetch_models_and_loras(function()
                update_dialog_status(dlg)
            end)
        end
    }
    
    dlg:separator{}
    
    -- Main prompt
    dlg:entry{
        id = "prompt",
        label = "Prompt:",
        text = current_settings.prompt,
        onchange = function()
            current_settings.prompt = dlg.data.prompt
        end
    }
    
    -- Quick presets
    dlg:combobox{
        id = "preset_prompts",
        label = "Presets:",
        options = preset_prompts,
        onchange = function()
            current_settings.prompt = dlg.data.preset_prompts
            dlg:modify{id = "prompt", text = current_settings.prompt}
        end
    }
    
    -- Size settings
    dlg:combobox{
        id = "dimension_presets",
        label = "Size:",
        options = (function()
            local options = {}
            for _, preset in ipairs(dimension_presets) do
                table.insert(options, preset.name)
            end
            return options
        end)(),
        onchange = function()
            local selected_name = dlg.data.dimension_presets
            for _, preset in ipairs(dimension_presets) do
                if preset.name == selected_name then
                    current_settings.pixel_width = preset.width
                    current_settings.pixel_height = preset.height
                    break
                end
            end
        end
    }
    
    dlg:number{
        id = "colors",
        label = "Colors:",
        text = tostring(current_settings.colors),
        decimals = 0,
        onchange = function()
            current_settings.colors = dlg.data.colors
        end
    }
    
    dlg:check{
        id = "remove_background",
        text = "Remove Background",
        selected = current_settings.remove_background,
        onclick = function()
            current_settings.remove_background = dlg.data.remove_background
        end
    }
    
    dlg:combobox{
        id = "output_method",
        label = "Output:",
        option = current_settings.output_method,
        options = {"New Layer", "New Frame"},
        onchange = function()
            current_settings.output_method = dlg.data.output_method
        end
    }
    
    dlg:separator{}
    
    -- Settings buttons
    dlg:button{text = "Model Settings", onclick = open_model_dialog}
    dlg:button{text = "Advanced Settings", onclick = open_advanced_dialog}
    
    dlg:separator{}
    
    -- Generation controls
    dlg:label{id="generation_time_label", text="Ready to generate"}
    
    dlg:button{
        id = "generate",
        text = "Generate Image",
        focus = true,
        onclick = function()
            -- Validation
            if current_settings.prompt == "" or current_settings.prompt == nil then
                app.alert("Please enter a prompt to generate an image.")
                return
            end

            if current_engine == "mlx" then
                app.alert("MLX engine placeholder: output will be a deterministic test pattern, not a real diffusion result.")
            end
            
            -- Start generation
            update_dialog_status(dlg)
            
            generate_image(current_settings, function(response, error)
                update_dialog_status(dlg)
                
                if error then
                    local emsg = tostring(error)
                    -- Fallback classification when server didn't respond with JSON
                    if emsg:find("INCOMPATIBLE_LORA") or emsg:find("size mismatch") then
                        app.alert("Selected LoRA is incompatible with the current base model. Try a different style or switch the base model.")
                    elseif emsg:lower():find("no base model") then
                        app.alert("No base model loaded. Open Model Settings and load one first.")
                    elseif emsg:lower():find("out of memory") then
                        app.alert("Out of memory. Choose Fast (512x512) or reduce steps/colors.")
                    else
                        app.alert("Generation failed. See server log for details.")
                    end
                elseif response and response.success and response.image then
                    place_image_in_aseprite_raw(response.image, current_settings.output_method)
                    
                    -- Show used seed but keep -1 for random generations
                    if response.seed and current_settings.seed == -1 then
                        app.alert("Image generated successfully!\nUsed seed: " .. response.seed .. "\n(Seed remains random for next generation)")
                    else
                        if current_engine == "mlx" then
                            app.alert("Placeholder image generated via MLX stub (not real model output).")
                        else
                            app.alert("Image generated successfully!")
                        end
                    end
                else
                    local code = response and response.code or "UNKNOWN_ERROR"
                    local map = {
                        INCOMPATIBLE_LORA = "Selected LoRA is incompatible with the current base model. Try a different style or switch the base model.",
                        NO_MODEL_LOADED = "No base model loaded. Open Model Settings and load one first.",
                        VALIDATION_ERROR = "Please enter a prompt before generating.",
                        OUT_OF_MEMORY = "Out of memory. Choose Fast (512x512) or reduce steps/colors.",
                        MODEL_LOAD_ERROR = "Model load failed. Check connection or offline mode and retry.",
                        MLX_NOT_INSTALLED = "MLX not installed. Run: pip install mlx mlx-lm (Apple Silicon only).",
                        MLX_NOT_IMPLEMENTED = "MLX backend not yet implemented for real generation.",
                        UNKNOWN_ERROR = response and response.error or "Generation failed. See server log for details." 
                    }
                    local friendly = map[code] or map.UNKNOWN_ERROR
                    app.alert(friendly)
                end
            end)
        end
    }
    
    dlg:button{text = "Close", onclick = function() dlg:close() end}
    
    -- Show dialog
    dlg:show{wait = false}
    
    -- Initial status update
    update_dialog_status(dlg)
end

-- Only run when explicitly called (not on script load)
print("Loading Local AI Generator v2.0...")
fetch_models_and_loras(create_main_dialog)