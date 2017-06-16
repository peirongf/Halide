#include "HalideRuntime.h"
#include "HalideBuffer.h"
#include "halide_benchmark.h"
#include "halide_image_io.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <vector>

extern "C" int halide_rungen_trampoline_argv(void **args);
extern "C" const struct halide_filter_metadata_t *halide_rungen_trampoline_metadata();

namespace {

using Halide::Runtime::Buffer;
using std::cerr;
using std::cout;
using std::istringstream;
using std::map;
using std::ostream;
using std::ostringstream;
using std::set;
using std::string;
using std::vector;

bool verbose = false;

ostream &operator<<(ostream &stream, const halide_type_t &type) {
    if (type.code == halide_type_uint && type.bits == 1) {
        stream << "bool";
    } else {
        switch (type.code) {
        case halide_type_int:
            stream << "int";
            break;
        case halide_type_uint:
            stream << "uint";
            break;
        case halide_type_float:
            stream << "float";
            break;
        case halide_type_handle:
            stream << "handle";
            break;
        default:
            stream << "#unknown";
            break;
        }
        stream << std::to_string(type.bits);
    }
    if (type.lanes > 1) {
        stream << ":" + std::to_string(type.lanes);
    }
    return stream;
}

ostream &operator<<(ostream &stream, const halide_dimension_t &d) {
    stream << "[" << d.min << "," << d.extent << "," << d.stride << "]";
    return stream;
}

ostream &operator<<(ostream &stream, const vector<halide_dimension_t> &shape) {
    stream << "[";
    bool need_comma = false;
    for (auto &d : shape) {
        if (need_comma) {
            stream << ',';
        }
        stream << d;
        need_comma = true;
    }
    stream << "]";
    return stream;
}

struct info {
    ostringstream msg;

    template<typename T>
    info &operator<<(const T &x) {
        if (verbose) {
            msg << x;
        }
        return *this;
    }

    ~info() {
        if (verbose) {
            cerr << msg.str();
            if (msg.str().back() != '\n') {
                cerr << '\n';
            }
        }
    }
};

struct warn {
    ostringstream msg;

    template<typename T>
    warn &operator<<(const T &x) {
        msg << x;
        return *this;
    }

    ~warn() {
        cerr << "Warning: " << msg.str();
        if (msg.str().back() != '\n') {
            cerr << '\n';
        }
    }
};

struct fail {
    ostringstream msg;

    template<typename T>
    fail &operator<<(const T &x) {
        msg << x;
        return *this;
    }

    ~fail() {
        cerr << msg.str();
        if (msg.str().back() != '\n') {
            cerr << '\n';
        }
        exit(1);
    }
};

extern "C" void halide_print(void *user_context, const char *message) {
    info() << "halide_print: " << message;
}

extern "C" void halide_error(void *user_context, const char *message) {
    fail() << "halide_error: " << message;
}

vector<string> split_string(const string &source, const string &delim) {
    vector<string> elements;
    size_t start = 0;
    size_t found = 0;
    while ((found = source.find(delim, start)) != string::npos) {
        elements.push_back(source.substr(start, found - start));
        start = found + delim.size();
    }

    // If start is exactly source.size(), the last thing in source is a
    // delimiter, in which case we want to add an empty string to elements.
    if (start <= source.size()) {
        elements.push_back(source.substr(start, string::npos));
    }
    return elements;
}

template<typename T>
bool parse_scalar(const string &str, T *t) {
    istringstream iss(str);
    iss >> *t;
    return(!iss.fail() && iss.get() == EOF);
}

template<>
bool parse_scalar(const string &str, bool *t) {
    if (str == "true") {
        *t = true;
        return true;
    } else if (str == "false") {
        *t = false;
        return true;
    } else {
        return false;
    }
}

template<>
bool parse_scalar(const string &str, void **t) {
    if (str == "nullptr") {
        *t = nullptr;
        return true;
    } else {
        return false;
    }
}

bool string_to_scalar(const halide_type_t &type,
                      const string &str,
                      halide_scalar_value_t *scalar) {
// can't use halide_type_t in a switch.
#define TYPE_AND_SIZE(CODE, BITS) (((CODE) << 8) | (BITS))
    switch (TYPE_AND_SIZE(type.code, type.bits)) {
    case TYPE_AND_SIZE(halide_type_uint, 1):
        return parse_scalar(str, &scalar->u.b);
    case TYPE_AND_SIZE(halide_type_float, 32):
        return parse_scalar(str, &scalar->u.f32);
    case TYPE_AND_SIZE(halide_type_float, 64):
        return parse_scalar(str, &scalar->u.f64);
    case TYPE_AND_SIZE(halide_type_int, 8):
        return parse_scalar(str, &scalar->u.i8);
    case TYPE_AND_SIZE(halide_type_int, 16):
        return parse_scalar(str, &scalar->u.i16);
    case TYPE_AND_SIZE(halide_type_int, 32):
        return parse_scalar(str, &scalar->u.i32);
    case TYPE_AND_SIZE(halide_type_int, 64):
        return parse_scalar(str, &scalar->u.i64);
    case TYPE_AND_SIZE(halide_type_uint, 8):
        return parse_scalar(str, &scalar->u.u8);
    case TYPE_AND_SIZE(halide_type_uint, 16):
        return parse_scalar(str, &scalar->u.u16);
    case TYPE_AND_SIZE(halide_type_uint, 32):
        return parse_scalar(str, &scalar->u.u32);
    case TYPE_AND_SIZE(halide_type_uint, 64):
        return parse_scalar(str, &scalar->u.u64);
    case TYPE_AND_SIZE(halide_type_handle, 64):
        return parse_scalar(str, &scalar->u.handle);
    default:
        break;
    }
    return false;
#undef TYPE_AND_SIZE
}

vector<halide_dimension_t> get_shape(const Buffer<> &b) {
    vector<halide_dimension_t> s;
    for (int i = 0; i < b.dimensions(); ++i) {
        s.push_back(b.raw_buffer()->dim[i]);
    }
    return s;
}


// BEGIN TODO: consider moving this to halide_image_io.h?
template<template<typename T> class Processor, typename... Args>
bool process_untyped_image(const halide_type_t &type, Args&&... args) {
// can't use halide_type_t in a switch.
#define TYPE_AND_SIZE(CODE, BITS) (((CODE) << 8) | (BITS))
    switch (TYPE_AND_SIZE(type.code, type.bits)) {
    case TYPE_AND_SIZE(halide_type_float, 32):
        return Processor<float>()(std::forward<Args>(args)...);
    case TYPE_AND_SIZE(halide_type_float, 64):
        return Processor<double>()(std::forward<Args>(args)...);
    case TYPE_AND_SIZE(halide_type_int, 8):
        return Processor<int8_t>()(std::forward<Args>(args)...);
    case TYPE_AND_SIZE(halide_type_int, 16):
        return Processor<int16_t>()(std::forward<Args>(args)...);
    case TYPE_AND_SIZE(halide_type_int, 32):
        return Processor<int32_t>()(std::forward<Args>(args)...);
    case TYPE_AND_SIZE(halide_type_int, 64):
        return Processor<int64_t>()(std::forward<Args>(args)...);
    case TYPE_AND_SIZE(halide_type_uint, 1):
        return Processor<bool>()(std::forward<Args>(args)...);
    case TYPE_AND_SIZE(halide_type_uint, 8):
        return Processor<uint8_t>()(std::forward<Args>(args)...);
    case TYPE_AND_SIZE(halide_type_uint, 16):
        return Processor<uint16_t>()(std::forward<Args>(args)...);
    case TYPE_AND_SIZE(halide_type_uint, 32):
        return Processor<uint32_t>()(std::forward<Args>(args)...);
    case TYPE_AND_SIZE(halide_type_uint, 64):
        return Processor<uint64_t>()(std::forward<Args>(args)...);
    case TYPE_AND_SIZE(halide_type_handle, 64):
        return Processor<void*>()(std::forward<Args>(args)...);
    default:
        cerr << "Unsupported type: " << type << "\n";
        return false;
    }
#undef TYPE_AND_SIZE
}

template <typename T>
struct load_untyped_image {
    bool operator()(const string &path, Buffer<> *image) {
        Buffer<T> tmp = Halide::Tools::load_image(path);
        *image = tmp;
        return true;
    }
};

// image_io won't compile for void*; specialize it away
template <>
struct load_untyped_image<void*> {
    bool operator()(const string &path, Buffer<> *image) {
        abort();
        return false;
    }
};

void load_image(const halide_type_t &type, const string &path, Buffer<> *image) {
    process_untyped_image<load_untyped_image>(type, path, image);
}

template <typename T>
struct save_untyped_image {
    bool operator()(const string &path, const Buffer<> &image) {
        Buffer<T> tmp = image;
        Halide::Tools::save_image(tmp, path);
        return true;
    }
};

// image_io won't compile for void*; specialize it away
template <>
struct save_untyped_image<void*> {
    bool operator()(const string &path, const Buffer<> &image) {
        abort();
        return false;
    }
};

void save_image(const halide_type_t &type, const string &path, const Buffer<> &image) {
    process_untyped_image<save_untyped_image>(type, path, image);
}
// END TODO: consider moving this to halide_image_io.h?

// BEGIN TODO: hacky algorithm inspired by Safelight
// (should really use the algorithm from AddImageChecks to come up with something more rigorous.)
vector<halide_dimension_t> choose_output_extents(int dimensions, const vector<halide_dimension_t> &defaults) {
    vector<halide_dimension_t> s(dimensions);
    for (int i = 0; i < dimensions; ++i) {
        if ((size_t) i < defaults.size()) {
            s[i] = defaults[i];
            continue;
        }
        s[i].extent = (i < 2 ? 1000 : 4);
    }
    return s;
}

vector<halide_dimension_t> fix_bounds_query_shape(const vector<halide_dimension_t> &constrained) {
    vector<halide_dimension_t> new_dims = constrained;

    // Make sure that the extents and strides for these are nonzero.
    for (size_t i = 0; i < new_dims.size(); ++i) {
        if (!new_dims[i].extent) {
            // A bit of a hack: fill in unconstrained dimensions to 1... except
            // for probably-the-channels dimension, which we'll special-case to
            // fill in to 4 when possible (unless it appears to be chunky).
            // Stride will be fixed below.
            if (i == 2) {
                if (constrained[0].stride >= 1 && constrained[2].stride == 1) {
                    // Definitely chunky, so make extent[2] match the chunk size
                    new_dims[i].extent = constrained[0].stride;
                } else {
                    // Not obviously chunky; let's go with 4 channels.
                    new_dims[i].extent = 4;
                }
            } else {
                new_dims[i].extent = 1;
            }
        }
    }

    // Special-case Chunky: most "chunky" generators tend to constrain stride[0]
    // and stride[2] to exact values, leaving stride[1] unconstrained;
    // in practice, we must ensure that stride[1] == stride[0] * extent[0]
    // and stride[0] = extent[2] to get results that are not garbled.
    // This is unpleasantly hacky and will likely need aditional enhancements.
    // (Note that there are, theoretically, other stride combinations that might
    // need fixing; in practice, ~all generators that aren't planar tend
    // to be classically chunky.)
    if (new_dims.size() >= 3) {
        if (constrained[2].stride == 1) {
            if (constrained[0].stride >= 1) {
                // If we have stride[0] and stride[2] set to obviously-chunky,
                // then force extent[2] to match stride[0].
                new_dims[2].extent = constrained[0].stride;
            } else {
                // If we have stride[2] == 1 but stride[0] <= 1,
                // force stride[0] = extent[2]
                new_dims[0].stride = new_dims[2].extent;
            }
            // Ensure stride[1] is reasonable.
            new_dims[1].stride = new_dims[0].extent * new_dims[0].stride;
        }
    }

    // If anything else is zero, just set strides to planar and hope for the best.
    bool zero_strides = false;
    for (size_t i = 0; i < new_dims.size(); ++i) {
        if (!new_dims[i].stride) {
            zero_strides = true;
        }
    }
    if (zero_strides) {
        // Planar
        new_dims[0].stride = 1;
        for (size_t i = 1; i < new_dims.size(); ++i) {
            new_dims[i].stride = new_dims[i - 1].stride * new_dims[i - 1].extent;
        }
    }
    return new_dims;
}
// END TODO: hacky algorithm inspired by Safelight

void usage() {
    cout << "TODO: usage instructions go here\n";
}

void do_describe(const halide_filter_metadata_t *md) {
    cout << "Filter name: \"" << md->name << "\"\n";
    for (size_t i = 0; i < (size_t) md->num_arguments; ++i) {
        auto &a = md->arguments[i];
        bool is_input = a.kind != halide_argument_kind_output_buffer;
        bool is_scalar = a.kind == halide_argument_kind_input_scalar;
        cout << "  " << (is_input ? "Input" : "Output") << " \"" << a.name << "\" is of type ";
        if (is_scalar) {
            cout << a.type;
        } else {
            cout << "Buffer<" << a.type << "> with " << a.dimensions << " dimensions";
        }
        cout << "\n";
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc <= 1) {
        usage();
        return 0;
    }

    const halide_filter_metadata_t *md = halide_rungen_trampoline_metadata();

    struct ArgData {
        size_t index{0};
        const halide_filter_argument_t* metadata{nullptr};
        string raw_string;
        halide_scalar_value_t scalar_value;
        Buffer<> buffer_value;
    };

    map<string, ArgData> args;
    set<string> found;
    for (size_t i = 0; i < (size_t) md->num_arguments; ++i) {
        ArgData arg;
        arg.index = i;
        arg.metadata = &md->arguments[i];
        if (arg.metadata->type.code == halide_type_handle) {
            // Pre-populate handle types with a default value of 'nullptr'
            // (the only legal value), so that they're ok to omit.
            arg.raw_string = "nullptr";
            found.insert(md->arguments[i].name);
        }
        args[md->arguments[i].name] = arg;
    }

    vector<halide_dimension_t> default_output_shape;
    vector<void*> filter_argv(md->num_arguments, nullptr);
    vector<string> unknown_args;
    bool benchmark = false;
    bool describe = false;
    int benchmark_samples = 3;
    int benchmark_iterations = 10;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            const char *p = argv[i] + 1; // skip -
            if (p[0] == '-') {
                p++; // allow -- as well, because why not
            }
            vector<string> v = split_string(p, "=");
            string flag_name = v[0];
            string flag_value = v.size() > 1 ? v[1] : "";
            if (v.size() > 2) {
                fail() << "Invalid argument: " << argv[i];
            }
            if (flag_name == "verbose") {
                if (flag_value.empty()) {
                    flag_value = "true";
                }
                if (!parse_scalar(flag_value, &verbose)) {
                    fail() << "Invalid value for flag: " << flag_name;
                }
            } else if (flag_name == "describe") {
                if (flag_value.empty()) {
                    flag_value = "true";
                }
                if (!parse_scalar(flag_value, &describe)) {
                    fail() << "Invalid value for flag: " << flag_name;
                }
            } else if (flag_name == "benchmark") {
                if (flag_value.empty()) {
                    flag_value = "true";
                }
                if (!parse_scalar(flag_value, &benchmark)) {
                    fail() << "Invalid value for flag: " << flag_name;
                }
            } else if (flag_name == "benchmark_samples") {
                if (!parse_scalar(flag_value, &benchmark_samples)) {
                    fail() << "Invalid value for flag: " << flag_name;
                }
            } else if (flag_name == "benchmark_iterations") {
                if (!parse_scalar(flag_value, &benchmark_iterations)) {
                    fail() << "Invalid value for flag: " << flag_name;
                }
            } else if (flag_name == "output_extents") {
                vector<string> extents = split_string(flag_value, ",");
                default_output_shape.clear();
                for (auto &s : extents) {
                    halide_dimension_t d = {0, 0, 0};
                    if (!parse_scalar(s, &d.extent)) {
                        fail() << "Invalid value for output_extent: " << s;
                    }
                    default_output_shape.push_back(d);
                }
            } else {
                usage();
                fail() << "Unknown flag: " << flag_name;
            }
        } else {
            // Assume it's a named Input or Output for the Generator,
            // in the form name=value.
            vector<string> v = split_string(argv[i], "=");
            if (v.size() != 2 || v[0].empty() || v[1].empty()) {
                fail() << "Invalid argument: " << argv[i];
            }
            const string &arg_name = v[0];
            const string &arg_value = v[1];
            if (args.find(arg_name) == args.end()) {
                // Gather up unknown-argument-names and show them
                // along with missing-argument-names, to make typos
                // easier to correct.
                unknown_args.push_back(arg_name);
                break;
            }
            if (arg_value.empty()) {
                fail() << "Argument value is empty for: " << arg_name;
            }
            auto &arg = args[arg_name];
            if (!arg.raw_string.empty()) {
                fail() << "Argument value specified multiple times for: " << arg_name;
            }
            arg.raw_string = arg_value;
            found.insert(arg_name);
        }
    }

    if (describe) {
        do_describe(md);
        return 0;
    }

    if (found.size() != args.size() || !unknown_args.empty()) {
        ostringstream o;
        for (auto &s : unknown_args) {
            o << "Unknown argument name: " << s << "\n";
        }
        for (auto &arg_pair : args) {
            auto &arg = arg_pair.second;
            if (arg.raw_string.empty()) {
                if (benchmark && arg.metadata->kind == halide_argument_kind_output_buffer) {
                    // It's OK to omit output arguments when we are benchmarking.
                    continue;
                }
                o << "Argument value missing for: " << arg.metadata->name << "\n";
            }
        }
        if (!o.str().empty()) {
            fail() << o.str();
        }
    }

    for (auto &arg_pair : args) {
        auto &arg_name = arg_pair.first;
        auto &arg = arg_pair.second;
        switch (arg.metadata->kind) {
        case halide_argument_kind_input_scalar: {
            if (!string_to_scalar(arg.metadata->type, arg.raw_string, &arg.scalar_value)) {
                fail() << "Argument value for: " << arg_name << " could not be parsed as type " 
                     << arg.metadata->type << ": " 
                     << arg.raw_string;
            }
            filter_argv[arg.index] = &arg.scalar_value;
            break;
        }
        case halide_argument_kind_input_buffer: {
            // TODO: add special paths to indicate generated random-noise, granger-rainbow, gradients, etc.,
            // and add relevant code to generate them
            info() << "Loading input " << arg_name << " from " << arg.raw_string << " ...";
            load_image(arg.metadata->type, arg.raw_string, &arg.buffer_value);
            const int dims_needed = arg.metadata->dimensions;
            const int dims_actual = arg.buffer_value.dimensions();
            if (dims_actual > dims_needed) {
                // If input has more dimensions than we need, trim off excess and issue a warning
                auto old_shape = get_shape(arg.buffer_value);
                while (arg.buffer_value.dimensions() > dims_needed) {
                    arg.buffer_value.slice(dims_needed, 0);
                }
                warn() << "Image loaded for argument " << arg_name << " has " 
                     << dims_actual << " dimensions, but this argument requires only "
                     << dims_needed << " dimensions";
                info() << "Shape for " << arg_name << " changed: " << old_shape << " -> " << get_shape(arg.buffer_value);
            } else if (dims_actual < dims_needed) {
                warn() << "Image loaded for argument " << arg_name << " has " 
                     << dims_actual << " dimensions, but this argument requires at least "
                     << dims_needed << " dimensions.";
                auto old_shape = get_shape(arg.buffer_value);
                while (arg.buffer_value.dimensions() < dims_needed) {
                    arg.buffer_value.embed(arg.buffer_value.dimensions(), 0);
                }
                info() << "Shape for " << arg_name << " changed: " << old_shape << " -> " << get_shape(arg.buffer_value);
            }
            // If there was no default_output_shape specified, use the shape of
            // the first input buffer (if any). 
            // TODO: this is often a better-than-nothing guess, but not always. Add a way to defeat it?
            if (default_output_shape.empty()) {
                default_output_shape = get_shape(arg.buffer_value);
            }
            filter_argv[arg.index] = arg.buffer_value.raw_buffer();
            break;
        }
        case halide_argument_kind_output_buffer:
            // Nothing yet
            break;
        }
    }

    {
        for (auto &arg_pair : args) {
            auto &arg = arg_pair.second;
            switch (arg.metadata->kind) {
            case halide_argument_kind_output_buffer:
                auto bounds_query_shape = choose_output_extents(arg.metadata->dimensions, default_output_shape);
                arg.buffer_value = Buffer<>(arg.metadata->type, nullptr, bounds_query_shape.size(), &bounds_query_shape[0]);
                filter_argv[arg.index] = arg.buffer_value.raw_buffer();
                break;
            }
        }

        info() << "Running bounds query...";
        int result = halide_rungen_trampoline_argv(&filter_argv[0]);
        if (result != 0) {
            fail() << "Bounds query failed with result code: " << result;
        }
    }

    {
        double pixels_out = 0.f;
        for (auto &arg_pair : args) {
            auto &arg_name = arg_pair.first;
            auto &arg = arg_pair.second;
            switch (arg.metadata->kind) {
            case halide_argument_kind_output_buffer:
                auto constrained_shape = get_shape(arg.buffer_value);
                info() << "Output " << arg_name << ": BoundsQuery result is " << constrained_shape;
                vector<halide_dimension_t> dims = fix_bounds_query_shape(constrained_shape);
                arg.buffer_value = Buffer<>(arg.metadata->type, nullptr, dims.size(), &dims[0]);
                info() << "Output " << arg_name << ": Shape is " << get_shape(arg.buffer_value);
                arg.buffer_value.check_overflow();
                arg.buffer_value.allocate();
                filter_argv[arg.index] = arg.buffer_value.raw_buffer();
                // TODO: this assumes that most output is "pixel-ish", and counting the size of the first
                // two dimensions approximates the "pixel size". This is not, in general, a valid assumption,
                // but is a useful metric for benchmarking.
                if (dims.size() >= 2) {
                    pixels_out += dims[0].extent * dims[1].extent;
                } else {
                    pixels_out += dims[0].extent;
                }
                break;
            }
        }

        if (benchmark) {
            // Run once to warm up cache. Ignore result since our halide_error() should catch everything.
            (void) halide_rungen_trampoline_argv(&filter_argv[0]);

            double time_in_seconds = Halide::Tools::benchmark(benchmark_samples, benchmark_iterations, [&filter_argv]() { 
                (void) halide_rungen_trampoline_argv(&filter_argv[0]);
            });
            double megapixels = pixels_out / (1024.0 * 1024.0);

            cout << "Benchmark for " << md->name << " produces best case of " << time_in_seconds << " sec/iter, over " 
                << benchmark_samples << " blocks of " << benchmark_iterations << " iterations.\n";
            cout << "Best output throughput is " << (megapixels / time_in_seconds) << " mpix/sec.\n";

            // TODO: add memory usage metrics (e.g. highwater)
        } else {
            info() << "Running filter...";
            int result = halide_rungen_trampoline_argv(&filter_argv[0]);
            if (result != 0) {
                fail() << "Filter failed with result code: " << result;
            }
        }
    }

    for (auto &arg_pair : args) {
        auto &arg_name = arg_pair.first;
        auto &arg = arg_pair.second;
        if (arg.metadata->kind == halide_argument_kind_output_buffer) {
            if (!arg.raw_string.empty()) {
                info() << "Saving output " << arg_name << " to " << arg.raw_string << " ...";
                // TODO: this will save output of float32 by coercing it to something else,
                // which may or may not be useful, but is always lossy. We should at least
                // emit warnings.
                save_image(arg.metadata->type, arg.raw_string, arg.buffer_value);
            } else {
                info() << "(Output " << arg_name << " was not saved.)";
            }
        }
    }

    return 0;
}
