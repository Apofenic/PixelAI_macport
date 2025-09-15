--[[
PixelAI Extension - Main Entry Point
=====================================

This is the primary entry point for the PixelAI Aseprite extension. It serves as a
lightweight wrapper that registers menu commands and delegates actual functionality
to the main UI script.

Purpose:
- Register extension commands in Aseprite's menu system
- Provide safe loading mechanism for the main UI components
- Enable hot-reload functionality for development
- Handle extension lifecycle (init/exit)

Architecture:
- Minimal main.lua keeps extension loading fast
- Actual functionality is in local-ui-main.lua for better organization
- Development commands allow testing without restarting Aseprite

Commands Added:
1. "Local AI Generator" - Main extension functionality
2. "Pixel AI Reload (Dev)" - Development hot-reload command
--]]

-- ============================================================================
-- DEBUG INSTRUMENTATION
-- ============================================================================
-- These debug prints help track extension loading during development
print("[PixelAI][DEBUG] main.lua loaded at runtime")
print("[PixelAI][DEBUG] Source path:", debug.getinfo(1, 'S').source)

-- ============================================================================
-- PATH RESOLUTION
-- ============================================================================
-- Determine the directory containing this script for loading other modules
local script_path = debug.getinfo(1, "S").source:sub(2)  -- Remove '@' prefix
local script_dir = script_path:match("(.*)[/\\]")        -- Extract directory path

-- ============================================================================
-- CORE FUNCTIONALITY
-- ============================================================================

--[[
Safely loads and executes the main UI script
This function provides error handling and debugging for the main functionality
--]]
local function run_ai_generator()
    -- Construct path to the main UI script
    local target = script_dir .. "/local-ui-main.lua"
    print("[PixelAI][DEBUG] Attempting to load UI file:", target)
    
    -- Use pcall for safe execution with error handling
    local success, result = pcall(dofile, target)
    if not success then
        -- Show user-friendly error message and log details
        app.alert("Error loading AI Generator: " .. tostring(result))
        print("[PixelAI][DEBUG] Load failure:", result)
    end
end

-- ============================================================================
-- EXTENSION LIFECYCLE
-- ============================================================================

--[[
Extension initialization function
Called by Aseprite when the extension is loaded

@param plugin: Plugin object provided by Aseprite for registering commands
--]]
function init(plugin)
    -- Register main extension command
    plugin:newCommand{
        id = "LocalAIGenerator",           -- Unique identifier for the command
        title = "Local AI Generator",      -- Display name in menus
        group = "file_scripts",            -- Menu group (affects placement)
        onclick = run_ai_generator         -- Function to execute when clicked
    }

    -- Register development hot-reload command
    -- This allows developers to test changes without restarting Aseprite
    plugin:newCommand{
        id = "PixelAIDevReload",
        title = "Pixel AI Reload (Dev)",
        group = "file_scripts",
        onclick = function()
            local target = script_dir .. "/local-ui-main.lua"
            print("[PixelAI][DEBUG] Dev reload -> " .. target)
            
            -- Attempt to reload the main UI script
            -- Note: This doesn't clear all state, but works for most development needs
            local ok, res = pcall(dofile, target)
            if not ok then
                app.alert("Reload failed: " .. tostring(res))
            else
                app.alert("PixelAI reloaded.")
            end
        end
    }
end

--[[
Extension cleanup function
Called by Aseprite when the extension is unloaded

@param plugin: Plugin object provided by Aseprite
Note: Currently no cleanup is needed, but this function must exist
--]]
function exit(plugin)
    -- No cleanup required for this extension
    -- The commands are automatically unregistered by Aseprite
end