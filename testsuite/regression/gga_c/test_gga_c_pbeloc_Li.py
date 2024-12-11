
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_pbeloc_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbeloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.844417944999992e-02, -4.894250604735734e-02, -1.655046134548394e-04, -1.604092499751150e-02, -6.549261789400851e-04, -5.096513489011301e-13, -3.333921680392926e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_pbeloc_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbeloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.232447941223294e-01, -1.231051106307007e-01, -1.205044395747396e-01, -1.203958077515403e-01, -1.704472838840914e-03, -1.704972193527598e-03, -2.416691613664718e-02, -1.129754568845439e-01, -5.155894041551187e-03, 2.795597855773414e-02, -5.662462592325266e-12, -5.688904982269580e-12, -3.325027834446756e-18, -4.463440115556536e-18])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_pbeloc_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_pbeloc", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.344012227110947e-05, 8.688024454221893e-05, 4.344012227110947e-05, 1.792483372116687e-04, 3.584966744233374e-04, 1.792483372116687e-04, 3.870045046384631e-04, 7.740090092769262e-04, 3.870045046384631e-04, 2.866350665941892e+00, 5.732701331883783e+00, 2.866350665941892e+00, 1.029992752913257e+01, 2.059985505826515e+01, 1.029992752913257e+01, 2.244510586218832e-08, 4.489021172437663e-08, 2.244510586218832e-08, 1.158809953359521e-10, 2.317619906719041e-10, 1.158809953359521e-10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
