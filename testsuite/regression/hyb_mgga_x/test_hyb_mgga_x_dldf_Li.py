
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_dldf_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-5.042949298022175e-01, -4.280917038359515e-01, -1.351755663812547e-01, -3.719187135335150e-02, -2.662405790016814e-02, -3.653283269816997e-02, -5.682599191323047e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_dldf_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.609948746120342e-01, -3.606740724846720e-01, -4.298521939088087e-01, -4.295644317310252e-01, -1.554343578679948e-01, -1.547346460142457e-01, -2.178216496198840e-02, -4.601761688034355e-02, -3.044221718347244e-02, -1.477697160990741e-03, -4.898274289706648e-02, -4.800190793026439e-02, -9.870480807997394e-04, -1.192814285701816e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_dldf_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.291486709725670e-04, 0.000000000000000e+00, -1.285774386085124e-04, -6.795020392322779e-04, 0.000000000000000e+00, -6.768596671326059e-04, -2.686312755532219e-01, 0.000000000000000e+00, -2.683583131666294e-01, -1.574158858407552e+00, 0.000000000000000e+00, -3.472543350877104e+00, -1.150977647622826e+02, 0.000000000000000e+00, -2.237382318893225e+01, -1.490123559845475e-03, 0.000000000000000e+00, -3.293848947142306e+00, -1.021242662172033e-09, 0.000000000000000e+00, -3.059668468647326e+12]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_dldf_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_dldf", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-4.426625941241902e-02, -4.434639280882865e-02, -2.058876264928946e-02, -2.067705205176741e-02, -4.556717338521426e-03, -4.697808168119023e-03, -7.911230737586384e-01, -3.603399102479492e-06, -4.233818891525599e-02, -7.372626452650272e-10, -1.789974594618776e-09, -3.889490668173692e-06, -1.002749366485596e-20, -1.398216616913095e-11]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
