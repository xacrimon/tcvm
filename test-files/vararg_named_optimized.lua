-- Named vararg, optimized path: `args` is only used as the base of
-- `t[exp]` / `t.id`, so no materialization is needed and the prologue
-- NOPs stay in place. VARARGGET dispatches on the nil sentinel.
local function get_by_index(...args)
    return args[1] + args[2] + args[3]
end

local function len(...args)
    return args.n
end

print(get_by_index(10, 20, 30))   -- 60
print(len(11, 22, 33, 44, 55))    -- 5
