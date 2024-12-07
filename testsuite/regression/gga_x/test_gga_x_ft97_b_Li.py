
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_ft97_b_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ft97_b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.788665298481220e+00, -1.278408096940544e+00, -4.689994942217263e-01, -1.597140804304665e-01, -8.411608302516130e-02, -1.416958137514063e-01, -5.549363010696781e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_ft97_b_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ft97_b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.244514804218471e+00, -2.246660697478437e+00, -1.512020266296748e+00, -1.513411383053372e+00, -2.867836328249905e-01, -2.867666540814990e-01, -2.055887635830687e-01, -4.128427773298325e-02, -6.296232835072547e-02, -8.315552029989718e-03, -4.237006295559853e-02, -4.252240592006448e-02, -7.999542530343944e-03, -6.905651943529886e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_ft97_b_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_ft97_b", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.388257967056570e-04, 0.000000000000000e+00, -2.379731231146106e-04, -1.010757547033005e-03, 0.000000000000000e+00, -1.007367592861403e-03, -1.641250213221142e-01, 0.000000000000000e+00, -1.640389267583093e-01, -3.587179051836116e+00, 0.000000000000000e+00, -1.393101311678555e+03, -1.076445457118075e+02, 0.000000000000000e+00, -5.007637181616992e+07, -1.211404126008542e+03, 0.000000000000000e+00, -1.213336625857558e+03, -1.484815949476714e+08, 0.000000000000000e+00, -4.420304095346343e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
