use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::time::{Duration, Instant};

#[derive(Debug, Default)]
struct InnerTimer {
    start: Option<Instant>,
    elapsed: Duration,
    subtimers: SubTimersMap,
}

impl InnerTimer {
    fn reset(&mut self) {
        self.start = None;
        self.elapsed = Duration::ZERO;
        self.subtimers.clear();
    }

    fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    fn stop(&mut self) {
        self.elapsed += self.start.unwrap().elapsed();
        self.start = None;
    }

    fn suspend(&mut self) {
        //save current elapsed and suspend
        //subtimers if this timer appears active
        if let Some(instant) = self.start {
            self.elapsed += instant.elapsed();
            self.subtimers.suspend();
        }
    }

    fn resume(&mut self) {
        //resume if this timer appears active.
        //just refresh start time to now.
        if self.start.is_some() {
            self.start = Some(Instant::now());
            self.subtimers.resume();
        }
    }

    fn elapsed(&self) -> Duration {
        self.elapsed
    }
}

#[derive(Debug, Default)]
struct SubTimersMap(HashMap<&'static str, InnerTimer>);

impl Deref for SubTimersMap {
    type Target = HashMap<&'static str, InnerTimer>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for SubTimersMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl SubTimersMap {
    fn reset_subtimer(&mut self, key: &'static str) {
        let t = self.entry(key).or_default();
        t.reset();
    }

    fn start_subtimer(&mut self, key: &'static str) {
        let t = self.0.entry(key).or_default();
        t.start();
    }

    #[allow(dead_code)]
    //not used but included for symmetry
    fn stop_subtimer(&mut self, key: &'static str) {
        let t = self.get_mut(key).unwrap();
        t.stop();
    }

    //this function suspends every timer in the
    //collection.   Used for notimeit!
    fn suspend(&mut self) {
        for t in self.values_mut() {
            t.suspend();
        }
    }

    fn resume(&mut self) {
        for t in self.values_mut() {
            t.resume();
        }
    }

    pub fn total_time(&self) -> Duration {
        self.values()
            .fold(Duration::ZERO, |acc, t| acc + t.elapsed())
    }

    pub fn print(&self, depth: u8) {
        for (key, val) in self.iter() {
            let tabs = format!("{: <1$}", "", 4 * depth as usize);
            println!("{}{:} : {:?}", tabs, *key, val.elapsed);
            val.subtimers.print(depth + 1);
        }
    }
}

#[derive(Default, Debug)]
pub struct Timers {
    stack: Vec<&'static str>,
    subtimers: SubTimersMap,
}

impl Timers {
    fn mut_active_timer(&mut self) -> Option<&mut InnerTimer> {
        if self.stack.is_empty() {
            return None;
        }

        //first one gets special treatment since self is not
        //an InnerTimer and a common trait would be overkill
        let key = &self.stack[0];
        let mut active_timer = self.subtimers.get_mut(key).unwrap();

        for key in self.stack.iter().skip(1) {
            active_timer = active_timer.subtimers.get_mut(key).unwrap();
        }
        Some(active_timer)
    }

    pub fn reset_timer(&mut self, key: &'static str) {
        self.subtimers.reset_subtimer(key);
    }

    pub fn start_as_current(&mut self, key: &'static str) {
        //starts a timer with name "str" as the current timer

        let active_timer = self.mut_active_timer();

        if let Some(active) = active_timer {
            // child of current active timer
            active.subtimers.start_subtimer(key);
        } else {
            // nothing active, create one at root
            self.subtimers.start_subtimer(key);
        }

        //append to timer call stack
        self.stack.push(key);
    }

    pub fn stop_current(&mut self) {
        //stops the current timer.  There should always be one
        // active when this function is reached.
        let active_timer = self.mut_active_timer();
        active_timer.unwrap().stop();

        //remove from timer call stack
        self.stack.pop();
    }

    //Suspend every timer in the collection.   Used for notimeit!
    pub fn suspend(&mut self) {
        self.subtimers.suspend();
    }

    //Resume every timer in the collection.   Used for notimeit!
    pub fn resume(&mut self) {
        self.subtimers.resume();
    }

    pub fn total_time(&self) -> Duration {
        self.subtimers.total_time()
    }

    pub fn print(&self) {
        self.subtimers.print(0);
    }
}

macro_rules! timeit {
    ($timer:ident => $key:literal; $($tt:tt)+) => {

        $timer.start_as_current($key);
        $(
            $tt
        )+
        $timer.stop_current();
    }
}
pub(crate) use timeit;

macro_rules! notimeit {
    ($timer:ident; $($tt:tt)+) => {

        $timer.suspend();
        $(
            $tt
        )+
        $timer.resume();
    }
}
pub(crate) use notimeit;
