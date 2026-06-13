include(joinpath(@__DIR__, "local_2d.jl"))

using Printf

const HYDRO_COLORS = (
    (68, 1, 84),
    (59, 82, 139),
    (33, 145, 140),
    (94, 201, 98),
    (253, 231, 37),
)

function color_hex(value, lo, hi)
    t = hi == lo ? 0.5 : clamp((value - lo) / (hi - lo), 0.0, 1.0)
    x = t * (length(HYDRO_COLORS) - 1) + 1
    i0 = floor(Int, x)
    i1 = min(i0 + 1, length(HYDRO_COLORS))
    f = x - i0

    c0 = HYDRO_COLORS[i0]
    c1 = HYDRO_COLORS[i1]
    r = round(Int, (1 - f) * c0[1] + f * c1[1])
    g = round(Int, (1 - f) * c0[2] + f * c1[2])
    b = round(Int, (1 - f) * c0[3] + f * c1[3])

    return @sprintf("#%02x%02x%02x", r, g, b)
end

function density_snapshot(u)
    return Array(interior_view(u[:rho]))
end

function pressure_snapshot(u; gamma=1.4)
    nx, ny = global_size(u[:rho])
    h = halo_width(u[:rho])
    data = field_storages(u)
    pressure = Matrix{Float64}(undef, nx, ny)

    @inbounds for I in CartesianIndices(interior_range(u[:rho]))
        i, j = Tuple(I)
        _, _, _, p, _ = primitive(conserved_cell(data, I), gamma)
        pressure[i - h, j - h] = p
    end

    return pressure
end

function write_heatmap!(io, field, x0, y0, width, height, title, lo, hi)
    nx, ny = size(field)
    dx = width / nx
    dy = height / ny

    @printf(io, "<text x=\"%.1f\" y=\"%.1f\" class=\"panel-title\">%s</text>\n",
        x0, y0 - 12, title)

    @inbounds for j in 1:ny, i in 1:nx
        x = x0 + (i - 1) * dx
        y = y0 + (ny - j) * dy
        fill = color_hex(field[i, j], lo, hi)
        @printf(io,
            "<rect x=\"%.3f\" y=\"%.3f\" width=\"%.3f\" height=\"%.3f\" fill=\"%s\"/>\n",
            x, y, dx + 0.05, dy + 0.05, fill)
    end

    @printf(io,
        "<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" class=\"panel-border\"/>\n",
        x0, y0, width, height)
    return nothing
end

function write_colorbar!(io, x0, y0, width, height, lo, hi)
    n = 80
    dy = height / n

    for k in 1:n
        t = (k - 1) / (n - 1)
        y = y0 + height - k * dy
        value = lo + t * (hi - lo)
        fill = color_hex(value, lo, hi)
        @printf(io,
            "<rect x=\"%.1f\" y=\"%.3f\" width=\"%.1f\" height=\"%.3f\" fill=\"%s\"/>\n",
            x0, y, width, dy + 0.05, fill)
    end

    @printf(io,
        "<rect x=\"%.1f\" y=\"%.1f\" width=\"%.1f\" height=\"%.1f\" class=\"panel-border\"/>\n",
        x0, y0, width, height)
    @printf(io, "<text x=\"%.1f\" y=\"%.1f\" class=\"tick\">%.4g</text>\n",
        x0 + width + 8, y0 + 4, hi)
    @printf(io, "<text x=\"%.1f\" y=\"%.1f\" class=\"tick\">%.4g</text>\n",
        x0 + width + 8, y0 + height, lo)
    return nothing
end

function write_hydro_svg(
        filename,
        initial_rho,
        final_rho,
        initial_pressure,
        final_pressure,
        info,
        initial,
        final,
)
    panel = 280.0
    gap = 34.0
    margin = 56.0
    colorbar_width = 18.0
    row_gap = 78.0
    header = 92.0
    footer = 52.0

    width = 2 * margin + 2 * panel + gap + colorbar_width + 82
    height = header + 2 * panel + row_gap + footer
    x1 = margin
    x2 = margin + panel + gap
    cbx = x2 + panel + 24
    y1 = header
    y2 = header + panel + row_gap

    rho_lo = min(minimum(initial_rho), minimum(final_rho))
    rho_hi = max(maximum(initial_rho), maximum(final_rho))
    pressure_lo = min(minimum(initial_pressure), minimum(final_pressure))
    pressure_hi = max(maximum(initial_pressure), maximum(final_pressure))

    open(filename, "w") do io
        @printf(io,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%.0f\" height=\"%.0f\" viewBox=\"0 0 %.0f %.0f\">\n",
            width, height, width, height)
        println(io, "<style>")
        println(io, "text { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #1f2933; }")
        println(io, ".title { font-size: 24px; font-weight: 700; }")
        println(io, ".subtitle { font-size: 13px; fill: #52606d; }")
        println(io, ".row-label { font-size: 15px; font-weight: 700; }")
        println(io, ".panel-title { font-size: 14px; font-weight: 600; }")
        println(io, ".tick { font-size: 11px; fill: #52606d; }")
        println(io, ".panel-border { fill: none; stroke: #1f2933; stroke-width: 1; }")
        println(io, "</style>")
        println(io, "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>")
        @printf(io, "<text x=\"%.1f\" y=\"36\" class=\"title\">Ideal hydro 2D: initial and final state</text>\n", margin)
        @printf(io,
            "<text x=\"%.1f\" y=\"60\" class=\"subtitle\">OrdinaryDiffEq Tsit5, adaptive=%s, CFL dtmax=%.3e, time=%.4f, grid=%dx%d</text>\n",
            margin, string(info.adaptive), info.dt, info.time, size(initial_rho, 1), size(initial_rho, 2))

        @printf(io, "<text x=\"%.1f\" y=\"%.1f\" class=\"row-label\">Density</text>\n", margin, y1 - 38)
        write_heatmap!(io, initial_rho, x1, y1, panel, panel, "initial rho", rho_lo, rho_hi)
        write_heatmap!(io, final_rho, x2, y1, panel, panel, "final rho", rho_lo, rho_hi)
        write_colorbar!(io, cbx, y1, colorbar_width, panel, rho_lo, rho_hi)

        @printf(io, "<text x=\"%.1f\" y=\"%.1f\" class=\"row-label\">Pressure</text>\n", margin, y2 - 38)
        write_heatmap!(io, initial_pressure, x1, y2, panel, panel, "initial p", pressure_lo, pressure_hi)
        write_heatmap!(io, final_pressure, x2, y2, panel, panel, "final p", pressure_lo, pressure_hi)
        write_colorbar!(io, cbx, y2, colorbar_width, panel, pressure_lo, pressure_hi)

        @printf(io,
            "<text x=\"%.1f\" y=\"%.1f\" class=\"subtitle\">mass error %.3e, energy error %.3e, final min rho %.6f, final min pressure %.6f</text>\n",
            margin, height - 22, final.mass - initial.mass, final.energy - initial.energy,
            final.min_rho, final.min_pressure)
        println(io, "</svg>")
    end

    return filename
end

function run_ideal_hydro_plot_2d(;
        nx=64,
        ny=64,
        steps=80,
        gamma=1.4,
        cfl=0.25,
        output=joinpath(tempdir(), "ideal_hydro_initial_final.svg"),
)
    u = ideal_hydro_state(nx, ny)
    fill_pressure_bump!(u; gamma)

    initial_rho = density_snapshot(u)
    initial_pressure = pressure_snapshot(u; gamma)
    dx = 1 / nx
    dy = 1 / ny
    initial = hydro_diagnostics(u; gamma, dx, dy)

    info = solve_ideal_hydro!(u; gamma, cfl, steps)
    final_rho = density_snapshot(u)
    final_pressure = pressure_snapshot(u; gamma)
    final = hydro_diagnostics(u; gamma, dx, dy)

    mkpath(dirname(output))
    write_hydro_svg(output, initial_rho, final_rho, initial_pressure, final_pressure, info, initial, final)
    return output, u, info, initial, final
end

function main()
    output = isempty(ARGS) ? joinpath(tempdir(), "ideal_hydro_initial_final.svg") : ARGS[1]
    filename, u, info, initial, final = run_ideal_hydro_plot_2d(; output)
    print_hydro_summary("LocalMultiHaloArray", u, info, initial, final)
    println("Wrote ", filename)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
