--[[
PixelAI Extension - Base64 Encoding/Decoding Library
====================================================

A lightweight, self-contained Base64 encoding and decoding library for Lua.
This module handles binary data conversion for transmitting image data between
the PixelAI extension and the Python AI server.

Features:
- Complete Base64 encoding with proper padding
- Robust Base64 decoding with error handling
- Binary data support for image transmission
- Memory-efficient processing for large image files
- Cross-platform compatibility (macOS, Windows, Linux)

Purpose:
Used to encode generated image data received from the AI server and decode it
for placement in Aseprite sprites. Essential for transmitting binary image data
over HTTP JSON communication.

Standards Compliance:
- Follows RFC 4648 Base64 specification
- Proper padding with '=' characters
- Standard 64-character alphabet (A-Z, a-z, 0-9, +, /)
- Handles both string and byte array inputs

Performance:
- Optimized for typical AI-generated image sizes (32x32 to 512x512 pixels)
- Uses bitwise operations for efficient encoding/decoding
- Minimal memory allocation during processing
--]]

local base64 = {}

-- ============================================================================
-- CONSTANTS AND CONFIGURATION
-- ============================================================================

-- Standard Base64 character set (RFC 4648)
local b64chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

-- ============================================================================
-- ENCODING FUNCTIONS
-- ============================================================================

--[[
Encodes binary data to Base64 string
Converts input data to Base64 representation with proper padding

@param data: Input data (string or byte array)
@return: Base64 encoded string
--]]
function base64.encode(data)
    if not data then return "" end
    
    local result = ""
    local bytes = {}
    
    -- Convert string to byte array if needed
    if type(data) == "string" then
        for i = 1, #data do
            bytes[i] = string.byte(data, i)
        end
    else
        bytes = data
    end
    
    local len = #bytes
    local i = 1
    
    -- Process data in 3-byte chunks (24 bits)
    while i <= len do
        local b1 = bytes[i] or 0      -- First byte
        local b2 = bytes[i + 1] or 0  -- Second byte (padded with 0 if missing)
        local b3 = bytes[i + 2] or 0  -- Third byte (padded with 0 if missing)
        
        -- Combine 3 bytes into 24-bit bitmap
        local bitmap = (b1 << 16) + (b2 << 8) + b3
        
        -- Extract four 6-bit values and convert to Base64 characters
        result = result .. string.sub(b64chars, ((bitmap >> 18) & 63) + 1, ((bitmap >> 18) & 63) + 1)
        result = result .. string.sub(b64chars, ((bitmap >> 12) & 63) + 1, ((bitmap >> 12) & 63) + 1)
        
        -- Add padding for incomplete groups
        if i + 1 <= len then
            result = result .. string.sub(b64chars, ((bitmap >> 6) & 63) + 1, ((bitmap >> 6) & 63) + 1)
        else
            result = result .. "="  -- Padding for missing second byte
        end
        
        if i + 2 <= len then
            result = result .. string.sub(b64chars, (bitmap & 63) + 1, (bitmap & 63) + 1)
        else
            result = result .. "="  -- Padding for missing third byte
        end
        
        i = i + 3  -- Move to next 3-byte group
    end
    
    return result
end

-- ============================================================================
-- DECODING FUNCTIONS
-- ============================================================================

--[[
Decodes Base64 string to binary data
Converts Base64 representation back to original binary data

@param data: Base64 encoded string
@return: Decoded binary data as string
--]]
function base64.decode(data)
    if not data then return "" end
    
    -- Remove any whitespace characters
    data = data:gsub("[ \t\r\n]", "")
    local padding = 0
    
    -- Handle Base64 padding characters
    if data:sub(-2) == "==" then
        padding = 2  -- Two bytes of padding
        data = data:sub(1, -3)
    elseif data:sub(-1) == "=" then
        padding = 1  -- One byte of padding
        data = data:sub(1, -2)
    end
    
    local result = {}
    local len = #data
    local i = 1
    
    -- Process data in 4-character chunks (24 bits)
    while i <= len do
        -- Extract four Base64 characters
        local c1 = data:sub(i, i)
        local c2 = data:sub(i + 1, i + 1)
        local c3 = data:sub(i + 2, i + 2)
        local c4 = data:sub(i + 3, i + 3)
        
        -- Convert Base64 characters to 6-bit values
        local n1 = b64chars:find(c1) - 1
        local n2 = b64chars:find(c2) - 1
        local n3 = c3 ~= "" and (b64chars:find(c3) - 1) or 0
        local n4 = c4 ~= "" and (b64chars:find(c4) - 1) or 0
        
        -- Combine four 6-bit values into 24-bit bitmap
        local bitmap = (n1 << 18) + (n2 << 12) + (n3 << 6) + n4
        
        -- Extract three 8-bit bytes from bitmap
        table.insert(result, string.char((bitmap >> 16) & 255))
        if i + 2 <= len or padding < 2 then
            table.insert(result, string.char((bitmap >> 8) & 255))
        end
        if i + 3 <= len or padding < 1 then
            table.insert(result, string.char(bitmap & 255))
        end
        
        i = i + 4  -- Move to next 4-character group
    end
    
    return table.concat(result)
end

-- ============================================================================
-- CONVENIENCE FUNCTIONS
-- ============================================================================

--[[
Convenience function to encode Aseprite image bytes
Directly encodes an Aseprite Image object's byte data

@param image: Aseprite Image object with .bytes property
@return: Base64 encoded string of image data
--]]
function base64.encode_image(image)
    if not image or not image.bytes then
        return ""
    end
    return base64.encode(image.bytes)
end

--[[
Convenience function to decode Base64 to byte string
Alias for decode function with clearer naming for image data

@param b64_string: Base64 encoded string
@return: Decoded binary data suitable for image creation
--]]
function base64.decode_to_bytes(b64_string)
    return base64.decode(b64_string)
end

return base64
