
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_r2scan01_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.600941601509476e-13, -2.102769347533950e-12, -2.886206898478072e-12, -1.284423956082748e-13, -1.473134853577189e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_r2scan01_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.035667527518117e-04, -2.199798756434693e-01, -3.934362875067394e-03, -2.101224685109989e-01, -8.079314398027879e-03, -1.812661508103900e-01, -3.451272109141609e-02, -1.052308408133263e-01, -1.084102153805201e-02, -1.251526061389837e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scan01_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.236210832608828e-03, 8.472421665217661e-03, 4.236210832608828e-03, 4.873258409991694e-03, 9.746516819983389e-03, 4.873258409991694e-03, 4.705249610570082e-02, 9.410499221140163e-02, 4.705249610570082e-02, 1.035933408273896e+01, 2.071866816547793e+01, 1.035933408273896e+01, 2.310086460299487e+04, 4.620172920598974e+04, 2.310086460299487e+04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_r2scan01_H_2_vtau():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_r2scan01", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.010329761587013e-02, -9.984851465454976e-03, -8.378541126812054e-03, -8.241961467583886e-03, -1.618704687779614e-02, -1.612823221291922e-02, -6.787824745357458e-02, -6.787648904233519e-02, -1.582634295797548e-02, -1.582634308136621e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
