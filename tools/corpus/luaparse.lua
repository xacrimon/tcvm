-- A Lua lexer and recursive-descent parser written in Lua: generate a large
-- source text, then tokenize and parse it into an AST N times, counting
-- the nodes.
local N = tonumber(arg and arg[1]) or 30

local byte, sub, find, char = string.byte, string.sub, string.find, string.char
local concat = table.concat

-- Lexer ---------------------------------------------------------------------

local keywords = {}
for w in ("and break do else elseif end false for function goto if in local nil not or repeat return then true until while"):gmatch("%a+") do
  keywords[w] = true
end

local B_NL, B_SPACE, B_TAB, B_CR = 10, 32, 9, 13
local B_0, B_9, B_DOT, B_MINUS = 48, 57, 46, 45
local B_QUOTE, B_DQUOTE, B_LBRACK, B_BACKSLASH = 39, 34, 91, 92

local function is_alpha(c)
  return c and (c >= 97 and c <= 122 or c >= 65 and c <= 90 or c == 95)
end
local function is_alnum(c)
  return c and (c >= 97 and c <= 122 or c >= 65 and c <= 90 or c == 95 or c >= 48 and c <= 57)
end

local three = {["..."] = true}
local two = {["=="] = true, ["~="] = true, ["<="] = true, [">="] = true, ["//"] = true,
  ["::"] = true, ["<<"] = true, [">>"] = true, [".."] = true}

-- Long bracket at `i` (`[`): its level, or nil.
local function long_level(src, i)
  local j = i + 1
  while byte(src, j) == 61 do j = j + 1 end
  if byte(src, j) == B_LBRACK then return j - i - 1 end
end

local function lex(src)
  local kinds, vals, lines = {}, {}, {}
  local n, i, line, len = 0, 1, 1, #src
  local function push(kind, val)
    n = n + 1
    kinds[n], vals[n], lines[n] = kind, val, line
  end
  while i <= len do
    local c = byte(src, i)
    if c == B_NL then
      line = line + 1
      i = i + 1
    elseif c == B_SPACE or c == B_TAB or c == B_CR then
      i = i + 1
    elseif c == B_MINUS and byte(src, i + 1) == B_MINUS then
      local lvl = byte(src, i + 2) == B_LBRACK and long_level(src, i + 2)
      if lvl then
        local close = "]" .. string.rep("=", lvl) .. "]"
        local _, e = find(src, close, i, true)
        for _ in sub(src, i, e):gmatch("\n") do line = line + 1 end
        i = e + 1
      else
        local e = find(src, "\n", i, true) or len + 1
        i = e
      end
    elseif is_alpha(c) then
      local j = i + 1
      while is_alnum(byte(src, j)) do j = j + 1 end
      local w = sub(src, i, j - 1)
      if keywords[w] then push(w, w) else push("name", w) end
      i = j
    elseif c >= B_0 and c <= B_9 or c == B_DOT and (byte(src, i + 1) or 0) >= B_0 and (byte(src, i + 1) or 0) <= B_9 then
      local s, e = find(src, "^0[xX][%x%.]+", i)
      if not s then s, e = find(src, "^[%d%.]+[eE][%+%-]?%d+", i) end
      if not s then s, e = find(src, "^[%d%.]+", i) end
      push("number", tonumber(sub(src, s, e)))
      i = e + 1
    elseif c == B_QUOTE or c == B_DQUOTE then
      local buf, j = {}, i + 1
      while true do
        local d = byte(src, j)
        if d == c then break end
        if d == B_BACKSLASH then
          local e = byte(src, j + 1)
          local map = {[110] = "\n", [116] = "\t", [114] = "\r", [92] = "\\", [34] = "\"", [39] = "'"}
          buf[#buf + 1] = map[e] or char(e)
          j = j + 2
        else
          local k = j
          while true do
            local x = byte(src, k)
            if x == c or x == B_BACKSLASH then break end
            k = k + 1
          end
          buf[#buf + 1] = sub(src, j, k - 1)
          j = k
        end
      end
      push("string", concat(buf))
      i = j + 1
    elseif c == B_LBRACK and long_level(src, i) then
      local lvl = long_level(src, i)
      local close = "]" .. string.rep("=", lvl) .. "]"
      local s, e = find(src, close, i, true)
      local body = sub(src, i + lvl + 2, s - 1)
      for _ in body:gmatch("\n") do line = line + 1 end
      push("string", body)
      i = e + 1
    else
      local op3, op2 = sub(src, i, i + 2), sub(src, i, i + 1)
      if three[op3] then push(op3, op3); i = i + 3
      elseif two[op2] then push(op2, op2); i = i + 2
      else
        local op = sub(src, i, i)
        push(op, op)
        i = i + 1
      end
    end
  end
  push("eof", nil)
  return kinds, vals, lines, n
end

-- Parser --------------------------------------------------------------------

local binprec = {
  ["or"] = {1, 1}, ["and"] = {2, 2},
  ["<"] = {3, 3}, [">"] = {3, 3}, ["<="] = {3, 3}, [">="] = {3, 3}, ["~="] = {3, 3}, ["=="] = {3, 3},
  ["|"] = {4, 4}, ["~"] = {5, 5}, ["&"] = {6, 6}, ["<<"] = {7, 7}, [">>"] = {7, 7},
  [".."] = {9, 8}, ["+"] = {10, 10}, ["-"] = {10, 10},
  ["*"] = {11, 11}, ["/"] = {11, 11}, ["//"] = {11, 11}, ["%"] = {11, 11},
  ["^"] = {14, 13},
}
local UNARY = 12

local function parse(kinds, vals, lines)
  local p = 1
  local function peek() return kinds[p] end
  local function take()
    local v = vals[p]
    p = p + 1
    return v
  end
  local function check(k)
    if kinds[p] ~= k then
      error(("line %d: '%s' expected near '%s'"):format(lines[p], k, tostring(vals[p])))
    end
    return take()
  end
  local function accept(k)
    if kinds[p] == k then p = p + 1 return true end
    return false
  end

  local block, expr

  local function funcbody(line)
    check("(")
    local params, vararg = {}, false
    if peek() ~= ")" then
      repeat
        if accept("...") then vararg = true break end
        params[#params + 1] = check("name")
      until not accept(",")
    end
    check(")")
    local body = block()
    check("end")
    return {tag = "Function", params = params, vararg = vararg, body = body, line = line}
  end

  local function exprlist()
    local list = {expr()}
    while accept(",") do list[#list + 1] = expr() end
    return list
  end

  local function table_cons()
    check("{")
    local items = {}
    while peek() ~= "}" do
      if peek() == "[" then
        take()
        local k = expr()
        check("]")
        check("=")
        items[#items + 1] = {tag = "Pair", key = k, value = expr()}
      elseif peek() == "name" and kinds[p + 1] == "=" then
        local k = take()
        take()
        items[#items + 1] = {tag = "Pair", key = {tag = "String", value = k}, value = expr()}
      else
        items[#items + 1] = expr()
      end
      if not accept(",") and not accept(";") then break end
    end
    check("}")
    return {tag = "Table", items = items}
  end

  local function args()
    local k = peek()
    if k == "string" then return {{tag = "String", value = take()}} end
    if k == "{" then return {table_cons()} end
    check("(")
    if accept(")") then return {} end
    local list = exprlist()
    check(")")
    return list
  end

  local function primary()
    local k = peek()
    if k == "name" then return {tag = "Name", name = take()} end
    check("(")
    local e = expr()
    check(")")
    return {tag = "Paren", expr = e}
  end

  local function suffixed()
    local e = primary()
    while true do
      local k = peek()
      if k == "." then
        take()
        e = {tag = "Index", obj = e, key = {tag = "String", value = check("name")}}
      elseif k == "[" then
        take()
        local key = expr()
        check("]")
        e = {tag = "Index", obj = e, key = key}
      elseif k == ":" then
        take()
        local name = check("name")
        e = {tag = "Invoke", obj = e, name = name, args = args()}
      elseif k == "(" or k == "string" or k == "{" then
        e = {tag = "Call", fn = e, args = args()}
      else
        return e
      end
    end
  end

  local function simple()
    local k = peek()
    if k == "number" then return {tag = "Number", value = take()} end
    if k == "string" then return {tag = "String", value = take()} end
    if k == "nil" then take() return {tag = "Nil"} end
    if k == "true" then take() return {tag = "True"} end
    if k == "false" then take() return {tag = "False"} end
    if k == "..." then take() return {tag = "Dots"} end
    if k == "{" then return table_cons() end
    if k == "function" then
      local line = lines[p]
      take()
      return funcbody(line)
    end
    return suffixed()
  end

  local function subexpr(limit)
    local e
    local k = peek()
    if k == "not" or k == "-" or k == "#" or k == "~" then
      take()
      e = {tag = "Unop", op = k, expr = subexpr(UNARY)}
    else
      e = simple()
    end
    while true do
      local op = peek()
      local prec = binprec[op]
      if not prec or prec[1] <= limit then return e end
      take()
      e = {tag = "Binop", op = op, left = e, right = subexpr(prec[2])}
    end
  end
  expr = function() return subexpr(0) end

  local function statement()
    local k = peek()
    local line = lines[p]
    if k == ";" then take() return nil end
    if k == "if" then
      take()
      local clauses = {}
      repeat
        local cond = expr()
        check("then")
        clauses[#clauses + 1] = {cond = cond, body = block()}
      until not accept("elseif")
      local orelse
      if accept("else") then orelse = block() end
      check("end")
      return {tag = "If", clauses = clauses, orelse = orelse, line = line}
    elseif k == "while" then
      take()
      local cond = expr()
      check("do")
      local body = block()
      check("end")
      return {tag = "While", cond = cond, body = body, line = line}
    elseif k == "do" then
      take()
      local body = block()
      check("end")
      return {tag = "Do", body = body}
    elseif k == "for" then
      take()
      local n1 = check("name")
      if accept("=") then
        local a = expr()
        check(",")
        local b = expr()
        local c
        if accept(",") then c = expr() end
        check("do")
        local body = block()
        check("end")
        return {tag = "Fornum", var = n1, from = a, to = b, step = c, body = body, line = line}
      end
      local names = {n1}
      while accept(",") do names[#names + 1] = check("name") end
      check("in")
      local exps = exprlist()
      check("do")
      local body = block()
      check("end")
      return {tag = "Forin", names = names, exps = exps, body = body, line = line}
    elseif k == "repeat" then
      take()
      local body = block()
      check("until")
      return {tag = "Repeat", body = body, cond = expr(), line = line}
    elseif k == "function" then
      take()
      local target = {tag = "Name", name = check("name")}
      local method = false
      while peek() == "." or peek() == ":" do
        local sep = take()
        local key = {tag = "String", value = check("name")}
        target = {tag = "Index", obj = target, key = key}
        if sep == ":" then method = true break end
      end
      local f = funcbody(line)
      f.method = method
      return {tag = "Set", targets = {target}, exps = {f}, line = line}
    elseif k == "local" then
      take()
      if accept("function") then
        local name = check("name")
        return {tag = "Localrec", name = name, func = funcbody(line), line = line}
      end
      local names = {}
      repeat
        names[#names + 1] = check("name")
        if accept("<") then check("name") check(">") end
      until not accept(",")
      local exps = {}
      if accept("=") then exps = exprlist() end
      return {tag = "Local", names = names, exps = exps, line = line}
    elseif k == "return" then
      take()
      local exps = {}
      local n = peek()
      if n ~= "end" and n ~= "else" and n ~= "elseif" and n ~= "until" and n ~= "eof" and n ~= ";" then
        exps = exprlist()
      end
      accept(";")
      return {tag = "Return", exps = exps, line = line}
    elseif k == "break" then
      take()
      return {tag = "Break"}
    elseif k == "goto" then
      take()
      return {tag = "Goto", label = check("name")}
    elseif k == "::" then
      take()
      local name = check("name")
      check("::")
      return {tag = "Label", name = name}
    end
    local e = suffixed()
    if peek() == "=" or peek() == "," then
      local targets = {e}
      while accept(",") do targets[#targets + 1] = suffixed() end
      check("=")
      return {tag = "Set", targets = targets, exps = exprlist(), line = line}
    end
    return {tag = "Callstat", call = e, line = line}
  end

  block = function()
    local stats = {}
    while true do
      local k = peek()
      if k == "end" or k == "else" or k == "elseif" or k == "until" or k == "eof" then break end
      local s = statement()
      if s then stats[#stats + 1] = s end
      if s and s.tag == "Return" then break end
    end
    return stats
  end

  local chunk = block()
  check("eof")
  return chunk
end

local function count(node)
  if type(node) ~= "table" then return 0 end
  local n = 1
  for _, v in pairs(node) do n = n + count(v) end
  return n
end

-- Source generator ----------------------------------------------------------

local function gen(nfuncs)
  local out = {}
  local seed = 12345
  local function rnd(k)
    seed = (seed * 1103515245 + 12345) % 2147483648
    return seed % k
  end
  local function e(d)
    local r = rnd(d > 2 and 4 or 9)
    if r == 0 then return tostring(rnd(1000)) end
    if r == 1 then return "x" .. rnd(5) end
    if r == 2 then return '"s' .. rnd(100) .. '\\n"' end
    if r == 3 then return "t[" .. e(d + 1) .. "]" end
    if r == 4 then return "f" .. rnd(nfuncs) .. "(" .. e(d + 1) .. ", " .. e(d + 1) .. ")" end
    if r == 5 then return "(" .. e(d + 1) .. " + " .. e(d + 1) .. ") * " .. e(d + 1) end
    if r == 6 then return "not " .. e(d + 1) .. " or " .. e(d + 1) .. " and " .. e(d + 1) end
    if r == 7 then return "{" .. e(d + 1) .. ", k = " .. e(d + 1) .. ", [" .. e(d + 1) .. "] = 1}" end
    return "obj:method" .. rnd(3) .. "(" .. e(d + 1) .. ").field .. 'x'"
  end
  for i = 0, nfuncs - 1 do
    out[#out + 1] = ("-- function %d\nlocal function f%d(x0, x1, ...)\n"):format(i, i)
    out[#out + 1] = "  local x2, x3, x4 = " .. e(1) .. ", " .. e(1) .. ", 0.5\n"
    for _ = 1, 6 do
      local r = rnd(5)
      if r == 0 then
        out[#out + 1] = "  for i = 1, " .. e(1) .. " do\n    x4 = x4 + " .. e(1) .. "\n  end\n"
      elseif r == 1 then
        out[#out + 1] = "  if " .. e(1) .. " then\n    x2 = " .. e(1) .. "\n  elseif x3 == 0x1F then\n    return " .. e(1) .. "\n  else\n    x3 = " .. e(1) .. "\n  end\n"
      elseif r == 2 then
        out[#out + 1] = "  for k, v in pairs(" .. e(1) .. ") do\n    t[k] = v .. " .. e(1) .. "\n  end\n"
      elseif r == 3 then
        out[#out + 1] = "  while x4 < 10 do\n    x4 = x4 + 1\n    print(" .. e(1) .. ")\n  end\n"
      else
        out[#out + 1] = "  local s = [[long\nstring]] .. " .. e(1) .. "\n  t.a.b[3] = s\n"
      end
    end
    out[#out + 1] = "  return x2, x3, " .. e(1) .. "\nend\n\n"
  end
  return concat(out)
end

local src = gen(400)
local total = 0
for _ = 1, N do
  local kinds, vals, lines, ntok = lex(src)
  local ast = parse(kinds, vals, lines)
  total = total + ntok + count(ast)
end
print(#src, total)
