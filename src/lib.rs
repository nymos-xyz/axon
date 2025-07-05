pub use axon_core as core;
pub use axon_social as social;
pub use axon_identity as identity;
pub use axon_contracts as contracts;
pub use axon_discovery as discovery;

pub mod prelude {
    pub use crate::core::*;
    pub use crate::social::*;
    pub use crate::identity::*;
    pub use crate::discovery::*;
}