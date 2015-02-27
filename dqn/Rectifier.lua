--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

--[[ Rectified Linear Unit.

The output is max(0, input).
--]]

local Rectifier, parent = torch.class('nn.Rectifier', 'nn.Module')

-- This module accepts minibatches
function Rectifier:updateOutput(input)
    return self.output:resizeAs(input):copy(input):abs():add(input):div(2)
end

function Rectifier:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(self.output)
    return self.gradInput:sign(self.output):cmul(gradOutput)
end