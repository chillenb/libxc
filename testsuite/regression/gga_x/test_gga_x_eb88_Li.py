
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_eb88_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_eb88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.799237929095074e+00, -1.287005750504451e+00, -4.288588795724324e-01, -1.604053559271746e-01, -8.069252657933322e-02, -1.331034729576340e-01, -5.361315485905242e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_eb88_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_eb88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.245056063580493e+00, -2.247176171287826e+00, -1.527102612813313e+00, -1.528464694372344e+00, -3.375243847605187e-01, -3.373915038008343e-01, -2.052505266792584e-01, -3.565803837898863e-02, -7.317623947116166e-02, -7.721829785838580e-03, -3.649186624566832e-02, -3.667032442651250e-02, -7.459777846191849e-03, -6.452113824581836e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_eb88_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_eb88", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.620368097613763e-04, 0.000000000000000e+00, -2.611665163278722e-04, -9.916531317493520e-04, 0.000000000000000e+00, -9.885367047800355e-04, -1.135944123425251e-01, 0.000000000000000e+00, -1.135648767575333e-01, -4.202198274325726e+00, 0.000000000000000e+00, -1.339916227333266e+03, -7.530541133271336e+01, 0.000000000000000e+00, -4.850334601161880e+07, -1.165169206508252e+03, 0.000000000000000e+00, -1.167020130806400e+03, -1.440007425326337e+08, 0.000000000000000e+00, -4.289622652402731e+08]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
