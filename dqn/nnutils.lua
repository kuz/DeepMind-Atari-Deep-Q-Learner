--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "torch"

function recursive_map(module, field, func)
    local str = ""
    if module[field] or module.modules then
        str = str .. torch.typename(module) .. ": "
    end
    if module[field] then
        str = str .. func(module[field])
    end
    if module.modules then
        str = str .. "["
        for i, submodule in ipairs(module.modules) do
            local submodule_str = recursive_map(submodule, field, func)
            str = str .. submodule_str
            if i < #module.modules and string.len(submodule_str) > 0 then
                str = str .. " "
            end
        end
        str = str .. "]"
    end

    return str
end

function abs_mean(w)
    return torch.mean(torch.abs(w:clone():float()))
end

function abs_max(w)
    return torch.abs(w:clone():float()):max()
end

-- Build a string of average absolute weight values for the modules in the
-- given network.
function get_weight_norms(module)
    return "Weight norms:\n" .. recursive_map(module, "weight", abs_mean) ..
            "\nWeight max:\n" .. recursive_map(module, "weight", abs_max)
end

-- Build a string of average absolute weight gradient values for the modules
-- in the given network.
function get_grad_norms(module)
    return "Weight grad norms:\n" ..
        recursive_map(module, "gradWeight", abs_mean) ..
        "\nWeight grad max:\n" .. recursive_map(module, "gradWeight", abs_max)
end
