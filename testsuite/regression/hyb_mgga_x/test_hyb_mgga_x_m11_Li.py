
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_m11_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-7.443232964408284e-01, -5.371049386222575e-01, -1.637897582223289e-01, -1.693942820248940e-02, -4.417812406293010e-03, -1.382231456138939e-04, -6.938117470463136e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_m11_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.600605428985188e-01, -3.551061207173382e-01, -7.081722494555148e-01, -7.106023955529646e-01, -1.817453248722467e-01, -1.768734144303571e-01, 1.138076309815719e-02, -2.384509504983219e-04, -4.342608193766270e-03, -7.724328347380880e-09, -2.794530425327657e-04, -2.709624647853187e-04, -2.301915033951880e-09, -5.598781261697521e-25]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m11_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.599432153391669e-04, 0.000000000000000e+00, 6.613447151466231e-04, 1.038788417665890e-03, 0.000000000000000e+00, 1.037635204071236e-03, 2.838571621095535e-02, 0.000000000000000e+00, 2.734227242747432e-02, 1.212248112944195e+01, 0.000000000000000e+00, -8.581386926439291e-03, -2.290218604312839e+01, 0.000000000000000e+00, -5.571325939642760e-05, -4.057139467802736e-06, 0.000000000000000e+00, -8.868283343561394e-03, -1.134548801656921e-15, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_m11_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_m11", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [-9.292797958186559e-02, -9.360467184439208e-02, -4.279811897232693e-03, -4.019276685865341e-03, -1.216354706901191e-02, -1.307142369319520e-02, -1.126974330322168e+00, -4.674625814326391e-08, -3.413202511571861e-02, -9.651019187359384e-15, -2.561915673589719e-11, -5.496870233118154e-08, -5.929193323922485e-26, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
