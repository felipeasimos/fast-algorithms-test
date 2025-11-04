const std = @import("std");

const GpuBackend = enum(u1) {
    wgpu,
    vulkan,
};

pub fn build(b: *std.Build) void {
    const use_llvm = b.option(bool, "use_llvm", "Use LLVM") orelse false;
    const gpu_backend = b.option(GpuBackend, "gpu_backend", "Which GPU backend to use") orelse .vulkan;

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const gpu_backend_name, const gpu_mod = gpu: {
        const gpu_backend_name = @tagName(gpu_backend);
        const gpu_mod = switch(gpu_backend) {
            .vulkan => b.dependency("vulkan", .{
                .registry = b.path("lib/vk.xml"),
                .video = b.path("lib/video.xml"),
            }).module("vulkan-zig"),
            .wgpu => b.dependency("wgpu_native_zig", .{
            }).module("wgpu"),
    };

    // library
    const lib = b.addModule("svdag", .{
        .root_source_file = b.path("src/svdag.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{
                .name = "vulkan",
                .module = gpu_mod,
            },
        },
    });

    // library tests
    const lib_tests_mod = b.createModule(.{
        .root_source_file = b.path("tests/tests.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "svdag", .module = lib },
        },
    });
    const lib_tests = b.addTest(.{
        .root_module = lib_tests_mod,
        .use_llvm = use_llvm,
    });
    const run_lib_tests = b.addRunArtifact(lib_tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_lib_tests.step);

    // executable
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "svdag", .module = lib },
        },
    });
    const exe = b.addExecutable(.{
        .name = "svdag",
        .root_module = exe_mod,
        .use_llvm = use_llvm,
    });
    b.installArtifact(exe);
    const run_exe = b.addRunArtifact(exe);

    const exe_step = b.step("run", "Run executable");
    exe_step.dependOn(&run_exe.step);

    // add check step for fast ZLS diagnostics on tests and library
    const check_lib_tests_mod = b.createModule(.{
        .root_source_file = b.path("tests/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    const check_lib_tests = b.addTest(.{
        .root_module = check_lib_tests_mod,
    });
    const check_exe = b.addExecutable(.{
        .name = "check_svdag",
        .root_module = exe_mod,
        .use_llvm = use_llvm,
    });
    const check_step = b.step("check", "Check for compile errors");
    check_step.dependOn(&check_lib_tests.step);
    check_step.dependOn(&check_exe.step);
}
