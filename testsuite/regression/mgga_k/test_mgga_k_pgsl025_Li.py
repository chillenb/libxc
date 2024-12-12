
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_k_pgsl025_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pgsl025", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([2.539529151491125e+02, 8.954800816783330e+00, 4.048534046493997e+00, 4.960569750578903e-01, 8.438251476892607e-02, 3.749315779897993e+05, 1.901137328563328e+15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_k_pgsl025_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pgsl025", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-3.711719806687757e+02, -3.704142545124585e+02, 9.661170991343306e+00, 9.683337268311581e+00, -4.611883796927473e+00, -4.567751304664217e+00, 2.004459797010856e-01, -1.789294365662294e+03, -6.999163130428147e-02, -7.299037550419751e+04, -1.234650240355403e+06, -1.617032112762536e+03, -4.306985078139690e+15, -2.221480941995506e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pgsl025_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pgsl025", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([5.749160568948524e-03, 0.000000000000000e+00, 5.727837265760318e-03, 2.580945326420985e-02, 0.000000000000000e+00, 2.572594671925097e-02, 4.147700292392334e+00, 0.000000000000000e+00, 4.153136729091924e+00, 5.871854119231491e+00, 0.000000000000000e+00, 7.829811571772351e+04, 4.018360953043848e+02, 0.000000000000000e+00, 2.454376727176486e+09, 6.733474755479385e+04, 0.000000000000000e+00, 6.882557364072798e+04, 8.236143114997760e+09, 0.000000000000000e+00, 2.292358375079883e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_k_pgsl025_Li_2_vlapl():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_k_pgsl025", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = numpy.asarray([-2.902562046858179e-01, -2.897354334130633e-01, -8.837855480261149e-03, -8.833313775854616e-03, 1.040205706378109e-01, 1.020580061075486e-01, -1.553109367464158e-02, 9.840321146500180e+01, 3.472270595989472e-02, 1.983347736593786e+04, 2.460219046558087e+03, 8.960042362283983e+01, 7.213021665955544e+09, 7.286824554258736e+04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
