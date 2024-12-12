
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_msb86bl_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.905352631571027e+00, -1.342261489445693e+00, -4.032349722417883e-01, -1.711367556258396e-01, -7.847771831744156e-02, -4.193712059716599e-02, -2.592067600503492e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_msb86bl_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.464662748233593e+00, -2.466975985512160e+00, -1.687744188579290e+00, -1.689439244453290e+00, -2.694129953041739e-01, -3.680507565505380e-01, -2.241701946059211e-01, -3.808704741161564e-02, -8.007562427869701e-02, -2.714981152054180e-03, -3.948513036864297e-02, -3.945567857480337e-02, -2.324628231899838e-03, -1.580401751801665e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_msb86bl_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-1.280821269465589e-04, 0.000000000000000e+00, -1.274970965291110e-04, -5.350037758423603e-04, 0.000000000000000e+00, -5.303249286676118e-04, -2.121936408085785e-01, 0.000000000000000e+00, -8.261582475674351e-02, -2.057759290554694e+00, 0.000000000000000e+00, -1.531959616143591e+02, -5.374612831600960e+01, 0.000000000000000e+00, -1.738393455310998e+06, -1.368383004951204e+02, 0.000000000000000e+00, -1.353797428809309e+02, -2.928593341034702e+06, 0.000000000000000e+00, -1.199821978092532e+07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_msb86bl_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_msb86bl", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([6.893428600843749e-06, 6.681881038898164e-12, 5.058347767793768e-05, 1.010046250929217e-17, 3.092602576943606e-02, 9.613179156520494e-11, 4.521303008837987e-03, 4.651148124961707e-15, 6.694166918747685e-07, 1.939396429554901e-04, 1.543562097618363e-18, 1.543396654983670e-15, 9.659766317434465e-31, 1.889546199712843e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
