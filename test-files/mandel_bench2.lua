local function mandel_pixel(x, y)
    -- configurables
    local width = 800
    local height = 800
    local max_iteration = 1000
    local x_min = -2.5
    local x_max = 1.0
    local y_min = -1.25
    local y_max = 1.25

    local cx = x_min + (x * (x_max - x_min) / width)
    local cy = y_min + (y * (y_max - y_min) / height)
    local zx = 0.0
    local zy = 0.0
    local iteration = 0

    while (zx * zx + zy * zy < 4.0) and (iteration < max_iteration) do
        local xtemp = zx * zx - zy * zy + cx
        zy = 2.0 * zx * zy + cy
        zx = xtemp
        iteration = iteration + 1
    end

    return iteration, zx, zy
end

-- Density ramp, sparse to solid; the last slot is reserved for points that
-- never escape (the interior of the set).
local ramp = " .,:;-~+=*oxOX%#&@"

local function mandel()
    -- configurables
    local width = 800
    local height = 800
    local max_iteration = 1000

    -- Visualization grid. Terminal cells are roughly twice as tall as they are
    -- wide, so the row step is doubled to keep the set from looking squashed.
    local cols = 100
    local rows = 40
    local step_x = width // cols
    local step_y = height // rows

    local levels = #ramp
    -- Escape counts pile up near zero, so a linear ramp would render nearly
    -- everything as blank; scale logarithmically instead.
    local log_max = math.log(max_iteration + 1)

    local total_iterations = 0
    local row = {}

    for y = 0, height - 1 do
        local sampled_row = y % step_y == 0
        local n = 0

        for x = 0, width - 1 do
            local iterations, zx, zy = mandel_pixel(x, y)
            total_iterations = total_iterations + iterations

            if sampled_row and x % step_x == 0 then
                local idx
                if iterations >= max_iteration then
                    idx = levels
                else
                    idx = math.floor(math.log(iterations + 1) / log_max * (levels - 1)) + 1
                end

                n = n + 1
                row[n] = string.sub(ramp, idx, idx)
            end
        end

        if sampled_row then
            print(table.concat(row, "", 1, n))
        end
    end

    return total_iterations
end

print(mandel())
