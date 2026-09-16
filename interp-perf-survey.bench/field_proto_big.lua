local Class = {}
Class.__index = Class
function Class.m0(s) return s.f0 end
function Class.m1(s) return s.f0 end
function Class.m2(s) return s.f0 end
function Class.m3(s) return s.f0 end
function Class.m4(s) return s.f0 end
function Class.m5(s) return s.f0 end
function Class.m6(s) return s.f0 end
function Class.m7(s) return s.f0 end
function Class.m8(s) return s.f0 end
function Class.m9(s) return s.f0 end
function Class.m10(s) return s.f0 end
function Class.m11(s) return s.f0 end
function Class.m12(s) return s.f0 end
function Class.m13(s) return s.f0 end
function Class.m14(s) return s.f0 end
function Class.m15(s) return s.f0 end
function Class.m16(s) return s.f0 end
function Class.m17(s) return s.f0 end
function Class.m18(s) return s.f0 end
function Class.m19(s) return s.f0 end
function Class.m20(s) return s.f0 end
function Class.m21(s) return s.f0 end
function Class.m22(s) return s.f0 end
function Class.m23(s) return s.f0 end
function Class.m24(s) return s.f0 end
function Class.m25(s) return s.f0 end
function Class.m26(s) return s.f0 end
function Class.m27(s) return s.f0 end
function Class.m28(s) return s.f0 end
function Class.m29(s) return s.f0 end
local obj = setmetatable({f0 = 0, f1 = 1, f2 = 2, f3 = 3, f4 = 4, f5 = 5, f6 = 6, f7 = 7, f8 = 8, f9 = 9, f10 = 10, f11 = 11}, Class)
local acc = 0
for i = 1, 30000000 do local f = obj.m29; if f then acc = acc + 1 end end
print(acc)
