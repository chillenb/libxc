
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_acggap_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_acggap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.162959935813039e-02, -4.551217750120349e-02, -4.269772803321074e-03, -1.571258051686454e-02, -2.402807427699548e-03, -3.720088347959616e-08, -1.161126316623351e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_acggap_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_acggap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.149988842171879e-01, -1.148714649075837e-01, -1.009962166662632e-01, -1.008994734764397e-01, -2.023071257860209e-02, -2.023816738992298e-02, -2.353618621140318e-02, -1.025061305996638e-01, -9.868276047176792e-03, 5.077130028670657e-01, -2.326969235528113e-07, -2.338786518765141e-07, -7.262778974146539e-15, -8.593745561089744e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_acggap_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_acggap", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.338785683147804e-05, 8.677571366295608e-05, 4.338785683147804e-05, 1.415804927246243e-04, 2.831609854492485e-04, 1.415804927246243e-04, 3.909435560773875e-03, 7.818871121547749e-03, 3.909435560773875e-03, 2.769236842139703e+00, 5.538473684279405e+00, 2.769236842139703e+00, 1.621344765041051e+01, 3.242689530082102e+01, 1.621344765041051e+01, 7.891449094256757e-04, 1.578289818851351e-03, 7.891449094256757e-04, 1.022750777008105e-05, 2.045501554016209e-05, 1.022750777008105e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
