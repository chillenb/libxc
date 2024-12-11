
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_tpss_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.346210087180268e-02, -8.371482262002407e-02, -4.959806172627841e-02, -1.808612568820412e-02, -1.095911360423736e-02, -5.892600477133273e-12, -8.300292557722951e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_tpss_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.026712119270149e-01, -1.025093296667022e-01, -9.254539378426754e-02, -9.240668603323736e-02, -5.664537600732751e-02, -5.668890672673786e-02, -2.101631195727098e-02, -1.243108863835546e-01, -1.310473963822716e-02, -7.152742107203552e-02, -4.266538431360686e-11, -3.375403352018220e-11, -1.026864148341892e-18, -7.910146580446758e-19])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpss_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.386252556484171e-04, 2.772505973492434e-04, 1.386252555248854e-04, 5.969146803740697e-04, 1.193829360748139e-03, 5.969146803740697e-04, 1.796789276683559e-01, 3.593578553367118e-01, 1.796789276683559e-01, 4.170868696918347e+00, 8.341737401120239e+00, 4.170869411439720e+00, 1.681681181378362e+02, 3.363362362756724e+02, 1.681681181378362e+02, 4.971943067632531e-09, 1.079340474789106e-08, 5.396702536288831e-09, 1.574454197631843e-15, -1.923003370021578e-15, -7.472775817567132e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_tpss_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_tpss", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-1.180368285926505e-09, -1.180368285926505e-09, -9.113724352682790e-88, -9.113724352682788e-88, -2.843802031518637e-80, -2.843802031518635e-80, -3.797175795639309e-10, -3.797175795638468e-10, -2.940009116187429e-25, -2.940009113812175e-25, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
