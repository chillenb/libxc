
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_scan_e0_H_2_zk():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_scan_e0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-3.209192242829097e-02, -2.146623832023406e-02, -1.427568895884108e-02, -4.770930985083006e-03, -1.715800527656941e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_scan_e0_H_2_vrho():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_scan_e0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.692621122888581e-02, 7.415270742969806e+00, -3.653576489694702e-02, 7.666490219011767e+01, -2.595917288710072e-02, 1.608118259395029e+01, -8.834986340322801e-03, -1.271208561786164e+00, -3.293877991411872e-04, 2.650979668268040e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_scan_e0_H_2_vsigma():
    # Prepare the input
    inp = test_data["H"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_scan_e0", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([1.518192902096407e-02, 3.036385804192813e-02, 1.518192902096407e-02, 6.309381046317875e-03, 1.261876209263575e-02, 6.309381046317875e-03, 2.334254299013846e-02, 4.668508598027692e-02, 2.334254299013846e-02, 4.002221699020060e-01, 8.004443398040121e-01, 4.002221699020060e-01, 9.189451247044151e+01, 1.837890249408830e+02, 9.189451247044151e+01])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
