#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/boost/graph/helpers.h>
#include <CGAL/IO/OBJ.h>
#include <CGAL/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/manifoldness.h>
#include <CGAL/Polygon_mesh_processing/merge_border_vertices.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace PMP = CGAL::Polygon_mesh_processing;

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Mesh = CGAL::Surface_mesh<Point_3>;

enum class BackendMode {
    AlphaWrap,
    Repair,
};

struct Options {
    std::string input_path;
    std::string output_path;
    BackendMode mode = BackendMode::AlphaWrap;
    double alpha = -1.0;
    double offset = -1.0;
    double alpha_relative = 1.0 / 50.0;
    double offset_relative = 1.0 / 30.0;
    bool repair_merge_boundary_vertices = true;
    bool repair_merge_reversible_components = true;
    bool repair_stitch_borders = true;
    bool repair_duplicate_non_manifold_vertices = true;
    bool verbose = true;
};

struct RunStats {
    std::size_t input_point_count = 0;
    std::size_t input_polygon_count = 0;
    std::size_t repaired_point_count = 0;
    std::size_t repaired_polygon_count = 0;
    bool soup_orientation_succeeded = false;
    double bbox_diagonal = 0.0;
    double alpha = 0.0;
    double offset = 0.0;
    std::size_t stitched_border_pairs = 0;
    std::size_t duplicated_non_manifold_vertices = 0;
    bool output_closed = false;
    bool output_self_intersecting = false;
};

[[noreturn]] void fail(const std::string& message)
{
    throw std::runtime_error(message);
}

void print_usage(const char* executable_name)
{
    std::cerr
        << "Usage: " << executable_name << " <input.obj> <output.obj> [options]\n"
        << "\n"
        << "Options:\n"
        << "  --mode <alpha-wrap|repair>  Backend mode. Default: alpha-wrap\n"
        << "  --alpha <value>             Absolute alpha value.\n"
        << "  --offset <value>            Absolute offset value.\n"
        << "  --alpha-relative <value>    Alpha as a fraction of the input bbox diagonal\n"
        << "                              when --alpha is omitted. Default: 0.02\n"
        << "  --offset-relative <value>   Offset as a fraction of alpha when --offset is\n"
        << "                              omitted. Default: 0.03333333333333333\n"
        << "  --skip-repair-merge-boundary-vertices\n"
        << "                              Disable merge_duplicated_vertices_in_boundary_cycles\n"
        << "                              in repair mode.\n"
        << "  --skip-repair-merge-reversible-components\n"
        << "                              Disable merge_reversible_connected_components in\n"
        << "                              repair mode.\n"
        << "  --skip-repair-stitch-borders\n"
        << "                              Disable stitch_borders in repair mode.\n"
        << "  --skip-repair-duplicate-non-manifold-vertices\n"
        << "                              Disable duplicate_non_manifold_vertices in repair\n"
        << "                              mode.\n"
        << "  --quiet                     Suppress progress output.\n";
}

BackendMode parse_mode(const std::string& value)
{
    if (value == "alpha-wrap") {
        return BackendMode::AlphaWrap;
    }
    if (value == "repair") {
        return BackendMode::Repair;
    }
    fail("unsupported mode: " + value + " (expected alpha-wrap or repair)");
}

const char* describe_mode(BackendMode mode)
{
    switch (mode) {
    case BackendMode::AlphaWrap:
        return "alpha-wrap";
    case BackendMode::Repair:
        return "repair";
    }
    return "unknown";
}

double parse_positive_double(const std::string& value, const std::string& flag_name)
{
    const double parsed = std::stod(value);
    if (!(parsed > 0.0)) {
        fail(flag_name + " must be greater than zero.");
    }
    return parsed;
}

Options parse_arguments(int argc, char** argv)
{
    if (argc < 3) {
        print_usage(argv[0]);
        fail("input and output OBJ paths are required.");
    }

    Options options;
    options.input_path = argv[1];
    options.output_path = argv[2];

    for (int index = 3; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--quiet") {
            options.verbose = false;
            continue;
        }
        if (argument == "--skip-repair-merge-boundary-vertices") {
            options.repair_merge_boundary_vertices = false;
            continue;
        }
        if (argument == "--skip-repair-merge-reversible-components") {
            options.repair_merge_reversible_components = false;
            continue;
        }
        if (argument == "--skip-repair-stitch-borders") {
            options.repair_stitch_borders = false;
            continue;
        }
        if (argument == "--skip-repair-duplicate-non-manifold-vertices") {
            options.repair_duplicate_non_manifold_vertices = false;
            continue;
        }
        if (index + 1 >= argc) {
            fail("missing value for argument: " + argument);
        }

        const std::string value = argv[++index];
        if (argument == "--mode") {
            options.mode = parse_mode(value);
        } else if (argument == "--alpha") {
            options.alpha = parse_positive_double(value, argument);
        } else if (argument == "--offset") {
            options.offset = parse_positive_double(value, argument);
        } else if (argument == "--alpha-relative") {
            options.alpha_relative = parse_positive_double(value, argument);
        } else if (argument == "--offset-relative") {
            options.offset_relative = parse_positive_double(value, argument);
        } else {
            fail("unsupported argument: " + argument);
        }
    }

    return options;
}

double compute_bbox_diagonal(const std::vector<Point_3>& points)
{
    if (points.empty()) {
        return 0.0;
    }

    double min_x = std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double min_z = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();

    for (const Point_3& point : points) {
        min_x = std::min(min_x, point.x());
        min_y = std::min(min_y, point.y());
        min_z = std::min(min_z, point.z());
        max_x = std::max(max_x, point.x());
        max_y = std::max(max_y, point.y());
        max_z = std::max(max_z, point.z());
    }

    const double dx = max_x - min_x;
    const double dy = max_y - min_y;
    const double dz = max_z - min_z;
    return std::sqrt((dx * dx) + (dy * dy) + (dz * dz));
}

void require_triangle_soup(const std::vector<std::vector<std::size_t>>& polygons)
{
    if (polygons.empty()) {
        fail("input polygon soup is empty after preprocessing.");
    }
    for (const auto& polygon : polygons) {
        if (polygon.size() != 3) {
            fail("input OBJ must contain only triangles for Alpha Wrap processing.");
        }
    }
}

void load_and_repair_triangle_soup(
    const Options& options,
    std::vector<Point_3>& points,
    std::vector<std::vector<std::size_t>>& polygons,
    RunStats& stats)
{
    const bool read_ok = CGAL::IO::read_OBJ(
        options.input_path,
        points,
        polygons,
        CGAL::parameters::verbose(options.verbose)
    );
    if (!read_ok) {
        fail("failed to read input OBJ: " + options.input_path);
    }
    if (points.empty() || polygons.empty()) {
        fail("input OBJ does not contain a usable polygon soup.");
    }

    stats.input_point_count = points.size();
    stats.input_polygon_count = polygons.size();

    PMP::repair_polygon_soup(points, polygons);
    require_triangle_soup(polygons);
    stats.soup_orientation_succeeded = PMP::orient_polygon_soup(points, polygons);
    stats.repaired_point_count = points.size();
    stats.repaired_polygon_count = polygons.size();
}

Mesh run_alpha_wrap_backend(
    const Options& options,
    const std::vector<Point_3>& points,
    const std::vector<std::vector<std::size_t>>& polygons,
    RunStats& stats)
{
    stats.bbox_diagonal = compute_bbox_diagonal(points);
    if (!(stats.bbox_diagonal > 0.0)) {
        fail("input mesh bounding box diagonal is zero.");
    }

    stats.alpha = options.alpha > 0.0 ? options.alpha : stats.bbox_diagonal * options.alpha_relative;
    stats.offset = options.offset > 0.0 ? options.offset : stats.alpha * options.offset_relative;
    if (!(stats.alpha > 0.0)) {
        fail("computed alpha must be greater than zero.");
    }
    if (!(stats.offset > 0.0)) {
        fail("computed offset must be greater than zero.");
    }

    Mesh wrapped_mesh;
    CGAL::alpha_wrap_3(
        points,
        polygons,
        stats.alpha,
        stats.offset,
        wrapped_mesh,
        CGAL::parameters::default_values(),
        CGAL::parameters::default_values()
    );
    if (num_faces(wrapped_mesh) == 0) {
        fail("Alpha Wrap produced an empty output mesh.");
    }

    stats.output_closed = CGAL::is_closed(wrapped_mesh);
    if (stats.output_closed) {
        PMP::orient_to_bound_a_volume(wrapped_mesh);
    }
    return wrapped_mesh;
}

Mesh run_repair_backend(
    const Options& options,
    const std::vector<Point_3>& points,
    const std::vector<std::vector<std::size_t>>& polygons,
    RunStats& stats)
{
    if (!PMP::is_polygon_soup_a_polygon_mesh(polygons)) {
        fail("CGAL repair mode could not convert the repaired polygon soup into a manifold polygon mesh. Try alpha-wrap for severely broken input.");
    }

    Mesh repaired_mesh;
    PMP::polygon_soup_to_polygon_mesh(points, polygons, repaired_mesh);
    if (num_faces(repaired_mesh) == 0) {
        fail("CGAL repair mode produced an empty output mesh.");
    }

    if (options.repair_merge_boundary_vertices) {
        PMP::merge_duplicated_vertices_in_boundary_cycles(repaired_mesh);
    }
    if (options.repair_merge_reversible_components) {
        PMP::merge_reversible_connected_components(repaired_mesh);
    }
    if (options.repair_stitch_borders) {
        stats.stitched_border_pairs = PMP::stitch_borders(repaired_mesh);
    }
    if (options.repair_duplicate_non_manifold_vertices) {
        stats.duplicated_non_manifold_vertices = PMP::duplicate_non_manifold_vertices(repaired_mesh);
    }
    stats.output_closed = CGAL::is_closed(repaired_mesh);
    stats.output_self_intersecting = PMP::does_self_intersect(repaired_mesh);

    if (stats.output_closed && !stats.output_self_intersecting) {
        PMP::orient_to_bound_a_volume(repaired_mesh);
    }

    return repaired_mesh;
}

int main(int argc, char** argv)
{
    try {
        const Options options = parse_arguments(argc, argv);

        std::vector<Point_3> points;
        std::vector<std::vector<std::size_t>> polygons;
        RunStats stats;
        load_and_repair_triangle_soup(options, points, polygons, stats);

        Mesh output_mesh;
        if (options.mode == BackendMode::AlphaWrap) {
            output_mesh = run_alpha_wrap_backend(options, points, polygons, stats);
        } else {
            output_mesh = run_repair_backend(options, points, polygons, stats);
        }

        const bool write_ok = CGAL::IO::write_polygon_mesh(
            options.output_path,
            output_mesh,
            CGAL::parameters::stream_precision(17).use_binary_mode(false)
        );
        if (!write_ok) {
            fail("failed to write output OBJ: " + options.output_path);
        }

        if (options.verbose) {
            std::cout
                << "mesh_heal_cgal_backend\n"
                << "mode=" << describe_mode(options.mode) << "\n"
                << "input_points=" << stats.input_point_count << "\n"
                << "input_polygons=" << stats.input_polygon_count << "\n"
                << "repaired_points=" << stats.repaired_point_count << "\n"
                << "repaired_polygons=" << stats.repaired_polygon_count << "\n"
                << "soup_orientation_succeeded=" << (stats.soup_orientation_succeeded ? "true" : "false") << "\n";

            if (options.mode == BackendMode::AlphaWrap) {
                std::cout
                    << "bbox_diagonal=" << stats.bbox_diagonal << "\n"
                    << "alpha=" << stats.alpha << "\n"
                    << "offset=" << stats.offset << "\n";
            } else {
                std::cout
                    << "repair_merge_boundary_vertices=" << (options.repair_merge_boundary_vertices ? "true" : "false") << "\n"
                    << "repair_merge_reversible_components=" << (options.repair_merge_reversible_components ? "true" : "false") << "\n"
                    << "repair_stitch_borders=" << (options.repair_stitch_borders ? "true" : "false") << "\n"
                    << "repair_duplicate_non_manifold_vertices=" << (options.repair_duplicate_non_manifold_vertices ? "true" : "false") << "\n"
                    << "stitched_border_pairs=" << stats.stitched_border_pairs << "\n"
                    << "duplicated_non_manifold_vertices=" << stats.duplicated_non_manifold_vertices << "\n"
                    << "output_closed=" << (stats.output_closed ? "true" : "false") << "\n"
                    << "output_self_intersecting=" << (stats.output_self_intersecting ? "true" : "false") << "\n";
            }

            std::cout
                << "output_vertices=" << num_vertices(output_mesh) << "\n"
                << "output_faces=" << num_faces(output_mesh) << "\n";
        }

        return EXIT_SUCCESS;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return EXIT_FAILURE;
    }
}