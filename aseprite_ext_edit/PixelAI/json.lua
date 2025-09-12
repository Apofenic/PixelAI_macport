-- Simple JSON library for Lua
-- Basic JSON encoding/decoding functionality

local json = {}

-- Helper function to escape strings
local function escape_string(str)
    str = string.gsub(str, "\\", "\\\\")
    str = string.gsub(str, '"', '\\"')
    str = string.gsub(str, "\n", "\\n")
    str = string.gsub(str, "\r", "\\r")
    str = string.gsub(str, "\t", "\\t")
    return str
end

-- Helper function to encode value
local function encode_value(value)
    local value_type = type(value)
    
    if value_type == "nil" then
        return "null"
    elseif value_type == "boolean" then
        return value and "true" or "false"
    elseif value_type == "number" then
        return tostring(value)
    elseif value_type == "string" then
        return '"' .. escape_string(value) .. '"'
    elseif value_type == "table" then
        -- Check if it's an array
        local is_array = true
        local max_index = 0
        local count = 0
        
        for k, v in pairs(value) do
            count = count + 1
            if type(k) == "number" and k > 0 and k == math.floor(k) then
                max_index = math.max(max_index, k)
            else
                is_array = false
                break
            end
        end
        
        if is_array and count == max_index then
            -- Encode as array
            local parts = {}
            for i = 1, max_index do
                table.insert(parts, encode_value(value[i]))
            end
            return "[" .. table.concat(parts, ",") .. "]"
        else
            -- Encode as object
            local parts = {}
            for k, v in pairs(value) do
                local key = type(k) == "string" and k or tostring(k)
                table.insert(parts, '"' .. escape_string(key) .. '":' .. encode_value(v))
            end
            return "{" .. table.concat(parts, ",") .. "}"
        end
    else
        error("Cannot encode value of type " .. value_type)
    end
end

-- JSON encode function
function json.encode(value)
    return encode_value(value)
end

-- Helper function to skip whitespace
local function skip_whitespace(str, pos)
    while pos <= #str do
        local char = string.sub(str, pos, pos)
        if char ~= " " and char ~= "\t" and char ~= "\n" and char ~= "\r" then
            break
        end
        pos = pos + 1
    end
    return pos
end

-- Helper function to parse string
local function parse_string(str, pos)
    if string.sub(str, pos, pos) ~= '"' then
        error("Expected '\"' at position " .. pos)
    end
    
    pos = pos + 1
    local result = ""
    
    while pos <= #str do
        local char = string.sub(str, pos, pos)
        
        if char == '"' then
            return result, pos + 1
        elseif char == "\\" then
            pos = pos + 1
            if pos > #str then
                error("Unexpected end of string")
            end
            
            local escape_char = string.sub(str, pos, pos)
            if escape_char == '"' then
                result = result .. '"'
            elseif escape_char == "\\" then
                result = result .. "\\"
            elseif escape_char == "/" then
                result = result .. "/"
            elseif escape_char == "b" then
                result = result .. "\b"
            elseif escape_char == "f" then
                result = result .. "\f"
            elseif escape_char == "n" then
                result = result .. "\n"
            elseif escape_char == "r" then
                result = result .. "\r"
            elseif escape_char == "t" then
                result = result .. "\t"
            else
                result = result .. escape_char
            end
        else
            result = result .. char
        end
        
        pos = pos + 1
    end
    
    error("Unterminated string")
end

-- Helper function to parse number
local function parse_number(str, pos)
    local start_pos = pos
    
    -- Handle negative sign
    if string.sub(str, pos, pos) == "-" then
        pos = pos + 1
    end
    
    -- Parse digits
    while pos <= #str and string.match(string.sub(str, pos, pos), "%d") do
        pos = pos + 1
    end
    
    -- Parse decimal part
    if pos <= #str and string.sub(str, pos, pos) == "." then
        pos = pos + 1
        while pos <= #str and string.match(string.sub(str, pos, pos), "%d") do
            pos = pos + 1
        end
    end
    
    -- Parse exponent
    if pos <= #str and (string.sub(str, pos, pos) == "e" or string.sub(str, pos, pos) == "E") then
        pos = pos + 1
        if pos <= #str and (string.sub(str, pos, pos) == "+" or string.sub(str, pos, pos) == "-") then
            pos = pos + 1
        end
        while pos <= #str and string.match(string.sub(str, pos, pos), "%d") do
            pos = pos + 1
        end
    end
    
    local number_str = string.sub(str, start_pos, pos - 1)
    return tonumber(number_str), pos
end

-- Forward declaration
local parse_value

-- Helper function to parse array
local function parse_array(str, pos)
    if string.sub(str, pos, pos) ~= "[" then
        error("Expected '[' at position " .. pos)
    end
    
    pos = pos + 1
    pos = skip_whitespace(str, pos)
    
    local result = {}
    
    if pos <= #str and string.sub(str, pos, pos) == "]" then
        return result, pos + 1
    end
    
    while true do
        local value
        value, pos = parse_value(str, pos)
        table.insert(result, value)
        
        pos = skip_whitespace(str, pos)
        
        if pos > #str then
            error("Unexpected end of array")
        end
        
        local char = string.sub(str, pos, pos)
        if char == "]" then
            return result, pos + 1
        elseif char == "," then
            pos = pos + 1
            pos = skip_whitespace(str, pos)
        else
            error("Expected ',' or ']' at position " .. pos)
        end
    end
end

-- Helper function to parse object
local function parse_object(str, pos)
    if string.sub(str, pos, pos) ~= "{" then
        error("Expected '{' at position " .. pos)
    end
    
    pos = pos + 1
    pos = skip_whitespace(str, pos)
    
    local result = {}
    
    if pos <= #str and string.sub(str, pos, pos) == "}" then
        return result, pos + 1
    end
    
    while true do
        pos = skip_whitespace(str, pos)
        
        local key
        key, pos = parse_string(str, pos)
        
        pos = skip_whitespace(str, pos)
        
        if pos > #str or string.sub(str, pos, pos) ~= ":" then
            error("Expected ':' at position " .. pos)
        end
        pos = pos + 1
        
        pos = skip_whitespace(str, pos)
        
        local value
        value, pos = parse_value(str, pos)
        
        result[key] = value
        
        pos = skip_whitespace(str, pos)
        
        if pos > #str then
            error("Unexpected end of object")
        end
        
        local char = string.sub(str, pos, pos)
        if char == "}" then
            return result, pos + 1
        elseif char == "," then
            pos = pos + 1
        else
            error("Expected ',' or '}' at position " .. pos)
        end
    end
end

-- Parse value function
parse_value = function(str, pos)
    pos = skip_whitespace(str, pos)
    
    if pos > #str then
        error("Unexpected end of input")
    end
    
    local char = string.sub(str, pos, pos)
    
    if char == '"' then
        return parse_string(str, pos)
    elseif char == "{" then
        return parse_object(str, pos)
    elseif char == "[" then
        return parse_array(str, pos)
    elseif char == "t" then
        if string.sub(str, pos, pos + 3) == "true" then
            return true, pos + 4
        else
            error("Invalid literal at position " .. pos)
        end
    elseif char == "f" then
        if string.sub(str, pos, pos + 4) == "false" then
            return false, pos + 5
        else
            error("Invalid literal at position " .. pos)
        end
    elseif char == "n" then
        if string.sub(str, pos, pos + 3) == "null" then
            return nil, pos + 4
        else
            error("Invalid literal at position " .. pos)
        end
    elseif char == "-" or string.match(char, "%d") then
        return parse_number(str, pos)
    else
        error("Unexpected character '" .. char .. "' at position " .. pos)
    end
end

-- JSON decode function
function json.decode(str)
    if type(str) ~= "string" then
        error("JSON decode expects a string")
    end
    
    local value, pos = parse_value(str, 1)
    pos = skip_whitespace(str, pos)
    
    if pos <= #str then
        error("Extra characters after JSON value")
    end
    
    return value
end

return json
