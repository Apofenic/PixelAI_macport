-- Extension wrapper for Local AI Generator
-- This just adds the button and loads the existing working script

-- DEBUG INSTRUMENTATION BEGIN
print("[PixelAI][DEBUG] main.lua loaded at runtime")
print("[PixelAI][DEBUG] Source path:", debug.getinfo(1, 'S').source)
-- DEBUG INSTRUMENTATION END

local script_path = debug.getinfo(1, "S").source:sub(2)
local script_dir = script_path:match("(.*)[/\\]")

-- Function to run the existing working script
local function run_ai_generator()
    -- Load and run the existing local-ui-main.lua script
    local target = script_dir .. "/local-ui-main.lua"
    print("[PixelAI][DEBUG] Attempting to load UI file:", target)
    local success, result = pcall(dofile, target)
    if not success then
        app.alert("Error loading AI Generator: " .. tostring(result))
        print("[PixelAI][DEBUG] Load failure:", result)
    end
end

-- Extension initialization - adds the button
function init(plugin)
    plugin:newCommand{
        id = "LocalAIGenerator",
        title = "Local AI Generator",
        group = "file_scripts",
        onclick = run_ai_generator
    }

    -- Development hot-reload command: re-executes local-ui-main.lua without reopening Aseprite
    plugin:newCommand{
        id = "PixelAIDevReload",
        title = "Pixel AI Reload (Dev)",
        group = "file_scripts",
        onclick = function()
            local target = script_dir .. "/local-ui-main.lua"
            print("[PixelAI][DEBUG] Dev reload -> " .. target)
            -- Clear any residual state (best effort) then reload
            local ok, res = pcall(dofile, target)
            if not ok then
                app.alert("Reload failed: " .. tostring(res))
            else
                app.alert("PixelAI reloaded.")
            end
        end
    }
end

-- Extension cleanup
function exit(plugin)
    -- Nothing to clean up
end