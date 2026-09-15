local function mandel_pixel(x, y)
    -- configurables
    local <const> width = 800
    local <const> height = 800
    local <const> max_iteration = 1000
    local <const> x_min = -2.5
    local <const> x_max = 1.0
    local <const> y_min = -1.25
    local <const> y_max = 1.25

    local <const> cx = x_min + (x * (x_max - x_min) / width)
    local <const> cy = y_min + (y * (y_max - y_min) / height)
    local zx = 0.0
    local zy = 0.0
    local iteration = 0

    while (zx * zx + zy * zy < 4.0) and (iteration < max_iteration) do
        local xtemp = zx * zx - zy * zy + cx
        zy = 2.0 * zx * zy + cy
        zx = xtemp
        iteration = iteration + 1
    end

    return iteration
end

local function mandel()
    -- configurables
    local <const> width = 800
    local <const> height = 800

    local total_iterations = 0

    for y = 0, height - 1 do
        for x = 0, width - 1 do
            total_iterations = total_iterations + mandel_pixel(x, y)
        end
    end

    return total_iterations
end

print(mandel())
