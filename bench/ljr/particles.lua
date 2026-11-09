-- "Typical" Lua benchmark: a small particle simulation written the way game
-- scripts usually are. Deliberately exercises what the numeric benches don't:
-- metatable method dispatch, per-call table allocation (GC churn), closures
-- with mutable upvalues, pairs/ipairs, dynamic string hash keys, and the
-- Lua<->native boundary (string.format, table.insert/remove/concat/sort).

local format, insert, remove, concat, sort = string.format, table.insert, table.remove, table.concat, table.sort
local floor = math.floor

-- Vec2 "class": every arithmetic op allocates a fresh table.
local Vec2 = {}
Vec2.__index = Vec2
function Vec2.new(x, y) return setmetatable({x = x, y = y}, Vec2) end
function Vec2:add(o) return Vec2.new(self.x + o.x, self.y + o.y) end
function Vec2:sub(o) return Vec2.new(self.x - o.x, self.y - o.y) end
function Vec2:scale(s) return Vec2.new(self.x * s, self.y * s) end
function Vec2:len2() return self.x * self.x + self.y * self.y end
Vec2.__add = Vec2.add
Vec2.__sub = Vec2.sub
Vec2.__tostring = function(v) return format("(%.2f, %.2f)", v.x, v.y) end

-- Deterministic LCG so the workload is identical across implementations.
local seed = 42
local function rand()
    seed = (seed * 1103515245 + 12345) % 2147483648
    return seed / 2147483648
end

-- Particle "class" with a per-instance update closure capturing p and drag.
local Particle = {}
Particle.__index = Particle
local next_id = 0
function Particle.new(pos, vel)
    next_id = next_id + 1
    local p = setmetatable({id = next_id, pos = pos, vel = vel, age = 0, alive = true}, Particle)
    local drag = 0.995
    p.update = function(dt)
        p.pos = p.pos + p.vel:scale(dt)
        p.vel = p.vel:scale(drag)
        p.age = p.age + 1
    end
    return p
end
function Particle:cell(size)
    return floor(self.pos.x / size), floor(self.pos.y / size)
end

local World = {}
World.__index = World
function World.new(n, size)
    local w = setmetatable({particles = {}, cell_size = size, frame = 0,
                            collisions = 0, spawned = 0, evicted = 0}, World)
    for _ = 1, n do w:spawn() end
    return w
end
function World:spawn()
    local pos = Vec2.new(rand() * 100, rand() * 100)
    local vel = Vec2.new(rand() * 2 - 1, rand() * 2 - 1)
    insert(self.particles, Particle.new(pos, vel))
    self.spawned = self.spawned + 1
end

-- Buckets keyed by a formatted "cx,cy" string: dynamic hash keys built at runtime.
function World:bucketize()
    local buckets = {}
    local size = self.cell_size
    for _, p in ipairs(self.particles) do
        local cx, cy = p:cell(size)
        local key = format("%d,%d", cx, cy)
        local b = buckets[key]
        if not b then
            b = {}
            buckets[key] = b
        end
        insert(b, p)
    end
    return buckets
end

function World:step(dt)
    self.frame = self.frame + 1
    for _, p in ipairs(self.particles) do p.update(dt) end

    local buckets = self:bucketize()
    local collisions = 0
    for _, b in pairs(buckets) do
        local n = #b
        for i = 1, n do
            local a = b[i]
            for j = i + 1, n do
                local d = a.pos - b[j].pos
                if d:len2() < 1.0 then
                    collisions = collisions + 1
                    a.alive = a.alive and (a.age < 200)
                end
            end
        end
    end
    self.collisions = self.collisions + collisions

    -- Evict old/dead particles from the back so table.remove stays cheap.
    local ps = self.particles
    for i = #ps, 1, -1 do
        local p = ps[i]
        if not p.alive or p.age > 400 then
            remove(ps, i)
            self.evicted = self.evicted + 1
        end
    end
    while #ps < 400 do self:spawn() end
end

function World:report()
    local ps = self.particles
    sort(ps, function(a, b) return a.id < b.id end)
    local lines = {}
    local sumx, sumy = 0, 0
    for _, p in ipairs(ps) do
        sumx, sumy = sumx + p.pos.x, sumy + p.pos.y
    end
    insert(lines, format("frame=%d particles=%d", self.frame, #ps))
    insert(lines, format("spawned=%d evicted=%d collisions=%d", self.spawned, self.evicted, self.collisions))
    insert(lines, format("centroid=%s", tostring(Vec2.new(sumx / #ps, sumy / #ps))))
    insert(lines, format("first=%d:%s last=%d:%s", ps[1].id, tostring(ps[1].pos), ps[#ps].id, tostring(ps[#ps].pos)))
    return concat(lines, "\n")
end

local frames = tonumber(arg and arg[1]) or 600
local world = World.new(400, 5)
for _ = 1, frames do world:step(0.1) end
print(world:report())
