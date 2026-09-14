local obj = {v = 5, inner = {}}
function obj:get() return self.v end
function obj:add(a, ...) return self.v + a, ... end
function obj.inner:m(y) return self, y end
function obj.inner.plain(y) return y end
