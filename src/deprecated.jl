# Deprecation shims for names renamed in 0.3.0. `@deprecate` re-exports the old
# name and emits a warning under `--depwarn=yes`; remove in 0.4.
#
#   get_send_view(s, d, u[, tile]) → edge_view(u, s, d[, tile])   (array first)
#   get_recv_view(s, d, u[, tile]) → ghost_view(u, s, d[, tile])
#   get_comm(u)                    → communicator(u)
#   isactive(x)                    → is_active(x)

@deprecate get_send_view(s::Side, d::Dim, u::AbstractHaloArray) edge_view(u, s, d)
@deprecate get_recv_view(s::Side, d::Dim, u::AbstractHaloArray) ghost_view(u, s, d)
@deprecate get_send_view(s::Side, d::Dim, u::ThreadedHaloArray, tile_id::Int) edge_view(u, s, d, tile_id)
@deprecate get_recv_view(s::Side, d::Dim, u::ThreadedHaloArray, tile_id::Int) ghost_view(u, s, d, tile_id)
@deprecate get_send_view(s::Side, d::Dim, arr::AbstractArray, halo::Int) edge_view(arr, s, d, halo)
@deprecate get_recv_view(s::Side, d::Dim, arr::AbstractArray, halo::Int) ghost_view(arr, s, d, halo)

@deprecate get_comm(u) communicator(u)
@deprecate isactive(x) is_active(x)
