function init(plugin)
    plugin:newCommand{
        id = "TestExtension",
        title = "Test Extension",
        group = "file_scripts",
        onclick = function()
            app.alert("Test extension is working!")
        end
    }
end

function exit(plugin)
end
