
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_mbrxc_bg_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxc_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.807858373520177e+00, -1.372652051618120e+00, -8.273854547169763e-01, -1.603706254293819e-01, -1.234351554599117e-01, -5.152580304038803e+00, -6.779253157273635e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_mbrxc_bg_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxc_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.033123526833603e+00, -2.035675547687101e+00, -1.207892171608644e+00, -1.209152821456342e+00, 1.235003348881779e-01, 1.253196339263754e-01, -1.968583934025398e-01, 5.472551312118537e+00, -2.067975107100289e-02, 4.575205251993853e+01, 5.196351528896307e+00, 5.406591099852199e+00, 7.582692142997942e+01, 8.715504466470570e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbrxc_bg_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxc_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-6.109396811384664e-04, 0.000000000000000e+00, -6.081010898591682e-04, -3.140542996813793e-03, 0.000000000000000e+00, -3.130457677045749e-03, -5.907582975499728e-01, 0.000000000000000e+00, -5.915718580871573e-01, -6.699155693561528e+00, 0.000000000000000e+00, -1.204030721171263e+05, -3.111130868916341e+02, 0.000000000000000e+00, -7.496071775716147e+10, -9.943153741740502e+04, 0.000000000000000e+00, -1.019818187002092e+05, -3.545739090028057e+11, 0.000000000000000e+00, -1.351358810444339e+12])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_mbrxc_bg_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_mbrxc_bg", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-2.833192737558196e-03, -2.829188857076617e-03, -3.744439943728193e-03, -3.743971981998104e-03, -1.439665962088745e-03, -1.434740230167297e-03, -2.624038946720086e-02, -2.507395974295670e-05, -1.436793327752939e-02, -1.265501265678814e-06, -2.684381386574132e-05, -2.565065856846721e-05, -6.790239928732763e-07, -5.622090820854007e-07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
