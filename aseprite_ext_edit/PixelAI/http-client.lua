
--[[
PixelAI Extension - HTTP Client Module
======================================

This module provides HTTP communication functionality for the PixelAI extension.
It handles both GET and POST requests to communicate with the local Python server
that runs the AI image generation.

Features:
- Dual-mode operation: Native Aseprite HTTP API with curl fallback
- Secure temporary file handling for large payloads
- macOS-optimized with absolute paths and proper redirection
- Extended timeouts for AI generation requests (up to 10 minutes)
- Robust error handling and JSON parsing
- Automatic cleanup of temporary files

Architecture:
- Uses Aseprite's native HTTP when available for better integration
- Falls back to system curl for reliability and extended timeout support
- Manages temporary files in system temp directory
- Provides consistent callback-based async interface

Dependencies:
- json.lua: For encoding/decoding JSON payloads
- System curl: For HTTP requests when native API insufficient
--]]

-- ============================================================================
-- INITIALIZATION AND DEPENDENCIES
-- ============================================================================

-- Get the script's directory to reliably load the JSON library
local script_path = debug.getinfo(1, "S").source:sub(2)
local script_dir = script_path:match("(.*)[/\\]")

-- Load the robust JSON library from the same directory
local json = dofile(script_dir .. "/json.lua")
if not json then
    app.alert("Critical Error: http-client.lua could not load json.lua. Ensure both files are in the same folder.")
    return
end

local http_client = {}

-- ============================================================================
-- PLATFORM CONFIGURATION
-- ============================================================================

-- macOS-only support with optimized paths and redirection
local is_macos = (package.config:sub(1,1) == '/')
local CURL = is_macos and '/usr/bin/curl' or 'curl.exe' -- Use absolute curl path on macOS
local DEVNULL = is_macos and '/dev/null' or 'nul'       -- Platform-specific null device

-- Ensure we're running on supported platform
if not is_macos then
    app.alert("This extension build is macOS-only. Please use the macOS version.")
end

-- ============================================================================
-- TEMPORARY FILE MANAGEMENT
-- ============================================================================

--[[
Gets the system temporary directory
Prefers TMPDIR on macOS, falls back to other common temp env vars

@return: Path to temporary directory
--]]
local function temp_dir()
    local td = os.getenv('TMPDIR') or os.getenv('TMP') or os.getenv('TEMP')
    if td and #td > 0 then return td end
    return is_macos and '/tmp' or '.'
end

-- Counter for generating unique temporary file names
local temp_counter = 0

--[[
Generates unique temporary filename
Uses timestamp, random number, and counter to ensure uniqueness

@return: Full path to temporary file
--]]
local function create_temp_filename()
    temp_counter = temp_counter + 1
    local timestamp = os.time()
    local random_part = math.random(1000, 9999)
    return string.format("%s/aseprite_temp_%d_%d_%d.tmp", temp_dir(), timestamp, random_part, temp_counter)
end

--[[
Creates temporary file with optional content
Used for both input data and output capture

@param content: Optional content to write to file
@return: Filename of created temporary file, or nil on failure
--]]
local function create_temp_file(content)
    local temp_name = create_temp_filename()
    if content then
        local file = io.open(temp_name, "wb")  -- Binary mode for cross-platform compatibility
        if file then
            file:write(content)
            file:close()
            return temp_name
        end
    else
        -- Just return the temp name for output file
        return temp_name
    end
    return nil
end

--[[
Reads entire file content safely
@param filename: Path to file to read
@return: File content as string, or nil on failure
--]]
local function read_file(filename)
    local file = io.open(filename, "rb")  -- Binary mode for consistent handling
    if file then
        local content = file:read("*all")
        file:close()
        return content
    end
    return nil
end

--[[
Safely removes temporary file
Uses pcall to prevent errors from breaking execution

@param filename: Path to file to remove
--]]
local function remove_file(filename)
    if filename then
        pcall(os.remove, filename)  -- Use pcall to avoid errors if file doesn't exist
    end
end

-- ============================================================================
-- HTTP REQUEST FUNCTIONS
-- ============================================================================

--[[
Performs HTTP GET request with dual-mode operation
Tries native Aseprite HTTP first, falls back to curl

@param url: URL to request
@param callback: Function to call with (response_data, error_message)
--]]
function http_client.get(url, callback)
    local temp_file = create_temp_file()
    if not temp_file then
        if callback then callback(nil, "Cannot create temporary file") end
        return
    end

    -- Try native Aseprite HTTP API first (better integration)
    local used_native = false
    if app and app.httpRequest then
        local ok, resp = pcall(app.httpRequest, {
            url = url,
            method = 'GET',
            timeout = 10  -- Short timeout for status checks
        })
        
        -- Check if native request succeeded with valid JSON response
        if ok and resp and resp.status and tostring(resp.status):match('^2') and resp.text then
            local success, parsed = pcall(json.decode, resp.text)
            if success then
                remove_file(temp_file)
                used_native = true
                if callback then callback(parsed, nil) end
                return
            end
        end
    end

    -- Fallback to curl if native API failed or unavailable
    if not used_native then
        local cmd
        if is_macos then
            cmd = string.format('%s -s -k --connect-timeout 5 --max-time 10 "%s" > "%s" 2>"%s"', 
                               CURL, url, temp_file, DEVNULL)
        else
            cmd = string.format('%s -s -k --connect-timeout 5 --max-time 10 "%s" > "%s" 2>%s', 
                               CURL, url, temp_file, DEVNULL)
        end
        os.execute(cmd)
    end

    -- Read response and parse JSON
    local content = read_file(temp_file)
    remove_file(temp_file)

    if content and content ~= "" then
        local success, parsed = pcall(json.decode, content)
        if success then
            if callback then callback(parsed, nil) end
        else
            if callback then callback(nil, "Failed to parse JSON response: " .. tostring(parsed)) end
        end
    else
        if callback then callback(nil, "Empty or no response from server") end
    end
end

--[[
Performs HTTP POST request with JSON payload
Uses extended timeout for AI generation requests

@param url: URL to post to
@param data: Lua table to encode as JSON and send
@param callback: Function to call with (response_data, error_message)
--]]
function http_client.post(url, data, callback)
    local temp_file = create_temp_file()
    local data_file = nil

    if not temp_file then
        if callback then callback(nil, "Cannot create temporary file") end
        return
    end

    -- Encode data as JSON and create temporary data file
    local json_data = json.encode(data)
    data_file = create_temp_file(json_data)
    if not data_file then
        remove_file(temp_file)
        if callback then callback(nil, "Cannot create data file") end
        return
    end

    -- Try native Aseprite HTTP API first
    if app and app.httpRequest then
        local ok, resp = pcall(app.httpRequest, {
            url = url,
            method = 'POST',
            headers = { ['Content-Type'] = 'application/json' },
            data = json_data,
            timeout = 600  -- Extended timeout for AI generation (10 minutes)
        })
        
        -- Check if native request succeeded
        if ok and resp and resp.status and tostring(resp.status):match('^2') and resp.text then
            local success, parsed = pcall(json.decode, resp.text)
            remove_file(temp_file)
            remove_file(data_file)
            if success then
                if callback then callback(parsed, nil) end
                return
            end
        end
        -- If native failed, fall through to curl
    end

    -- *** EXTENDED TIMEOUT: 600 seconds (10 minutes) for AI generation ***
    -- This accommodates the time needed for complex AI image generation
    local cmd
    if is_macos then
        cmd = string.format('%s -s -k --connect-timeout 10 --max-time 600 -X POST -H "Content-Type: application/json" -d @"%s" "%s" > "%s" 2>"%s"',
                            CURL, data_file, url, temp_file, DEVNULL)
    else
        cmd = string.format('%s -s -k --connect-timeout 10 --max-time 600 -X POST -H "Content-Type: application/json" -d @"%s" "%s" > "%s" 2>%s',
                            CURL, data_file, url, temp_file, DEVNULL)
    end

    os.execute(cmd)

    -- Read response and clean up temporary files
    local content = read_file(temp_file)
    remove_file(temp_file)
    remove_file(data_file)

    if content and content ~= "" then
        local success, parsed = pcall(json.decode, content)
        if success then
            if callback then callback(parsed, nil) end
        else
            if callback then callback(nil, "Failed to parse JSON response: " .. tostring(parsed)) end
        end
    else
        if callback then callback(nil, "Empty or no response from server") end
    end
end

return http_client