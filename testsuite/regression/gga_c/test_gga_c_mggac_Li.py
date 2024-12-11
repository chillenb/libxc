
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_mggac_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_mggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-7.390270611555647e-02, -5.889447313733547e-02, -9.346162544079918e-03, -1.669553603342114e-02, -3.551262488419978e-03, -3.759247183248576e-08, -8.886934157320558e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_mggac_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_mggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.168034708859704e-01, -1.166594546937935e-01, -1.069779316219968e-01, -1.068623025479125e-01, -3.895692390654775e-02, -3.897186943717683e-02, -2.272381837923093e-02, -1.126402800705011e-01, -1.307169735208390e-02, 5.570121941387048e-01, -2.431600371618455e-07, -2.443817348432330e-07, -5.621649250709543e-15, -6.651695524679396e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_mggac_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_mggac", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([3.302116219908766e-05, 6.604232439817530e-05, 3.302116219908766e-05, 1.182391988884573e-04, 2.364783977769147e-04, 1.182391988884573e-04, 7.153413142202885e-03, 1.430682628440577e-02, 7.153413142202885e-03, 1.707655292277012e+00, 3.415310584554024e+00, 1.707655292277012e+00, 2.011198690447692e+01, 4.022397380895385e+01, 2.011198690447692e+01, 8.286647563037904e-04, 1.657329512611567e-03, 8.286647563037904e-04, 7.947430011032487e-06, 1.589505556088007e-05, 7.947430011032487e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
