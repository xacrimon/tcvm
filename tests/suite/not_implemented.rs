//! Library functions tcvm doesn't implement yet raise an error naming the
//! function, instead of panicking.

use crate::common::ok;

#[test]
fn stubs_raise() {
    assert_eq!(
        ok("local out = {}
            for _, name in ipairs{'debug.getinfo', 'debug.sethook', 'require',
                                  'package.searchpath', 'os.execute', 'io.popen',
                                  'string.dump'} do
                local f = load('return ' .. name)()
                local ok, e = pcall(f)
                out[#out + 1] = tostring(ok) .. ' ' .. e
            end
            return table.concat(out, '\\n')"),
        "false debug.getinfo is not implemented\n\
         false debug.sethook is not implemented\n\
         false require is not implemented\n\
         false package.searchpath is not implemented\n\
         false os.execute is not implemented\n\
         false io.popen is not implemented\n\
         false string.dump is not implemented"
    );
}

/// `debug.traceback` hands its message back, without the traceback lua adds.
#[test]
fn traceback_returns_the_message() {
    assert_eq!(
        ok("local t = {}
            return cat(select(2, xpcall(function() error('x') end, debug.traceback)),
                       debug.traceback(42), debug.traceback(t) == t, debug.traceback(nil))"),
        "c:2: x 42 true nil"
    );
}
