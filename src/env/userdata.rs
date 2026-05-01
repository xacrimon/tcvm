use crate::dmm::{Collect, Gc, Mutation, RefLock};
use crate::env::table::Table;
use crate::env::value::Value;

/// Copy wrapper stored in Value.
#[derive(Clone, Copy, Collect)]
#[collect(internal, no_drop)]
pub struct Userdata<'gc>(Gc<'gc, RefLock<UserdataState<'gc>>>);

#[derive(Collect)]
#[collect(internal, no_drop)]
pub struct UserdataState<'gc> {
    #[collect(require_static)]
    data: Box<dyn std::any::Any>,
    user_values: Vec<Value<'gc>>,
    metatable: Option<Table<'gc>>,
}

impl<'gc> Userdata<'gc> {
    pub fn new<T: 'static>(mc: &Mutation<'gc>, data: T, num_user_values: usize) -> Self {
        let state = UserdataState {
            data: Box::new(data),
            user_values: vec![Value::nil(); num_user_values],
            metatable: None,
        };
        Userdata(Gc::new(mc, RefLock::new(state)))
    }

    pub fn metatable(self) -> Option<Table<'gc>> {
        self.0.borrow().metatable
    }

    pub fn set_metatable(self, mc: &Mutation<'gc>, mt: Option<Table<'gc>>) {
        self.0.borrow_mut(mc).metatable = mt;
    }

    pub fn get_user_value(self, index: usize) -> Value<'gc> {
        self.0
            .borrow()
            .user_values
            .get(index)
            .copied()
            .unwrap_or(Value::nil())
    }

    pub fn set_user_value(self, mc: &Mutation<'gc>, index: usize, value: Value<'gc>) {
        self.0.borrow_mut(mc).user_values[index] = value;
    }

    pub fn inner(self) -> Gc<'gc, RefLock<UserdataState<'gc>>> {
        self.0
    }

    pub(crate) fn from_inner(g: Gc<'gc, RefLock<UserdataState<'gc>>>) -> Self {
        Userdata(g)
    }
}
