
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_c_rppscan_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.100620815443640e-02, -5.629325295468334e-02, -6.498879409859544e-02, -2.344945833628993e-03, -1.518603161346362e-02, 1.586474458610381e+02, 1.936214857031430e+09])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_c_rppscan_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.344021055912601e-02, -1.329224088507735e-02, 1.051458023500490e-03, 1.263384416414221e-03, -6.820947134485039e-02, -6.826684027960464e-02, 1.780108697932523e-03, -1.571689213660388e-01, -1.229191849437274e-02, -5.721243928705702e-02, -9.376477001260062e+04, -9.376487533400470e+04, -2.480413813549017e+10, -2.482520841340990e+10])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rppscan_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([7.210467105336919e-05, 1.442093421067384e-04, 7.210467105336919e-05, 5.730802292815383e-04, 1.146160458563077e-03, 5.730802292815383e-04, 2.484573062997976e-01, 4.969146125995952e-01, 2.484573062997976e-01, 3.589510744654830e+00, 7.179021489309659e+00, 3.589510744654830e+00, 1.901619938335185e+02, 3.803239876670371e+02, 1.901619938335185e+02, 4.138445724759062e+07, 8.276891449518125e+07, 4.138445724759062e+07, 8.794345467502752e+14, 1.758869093500550e+15, 8.794345467502752e+14])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_c_rppscan_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_c_rppscan", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([-4.266236406216519e-03, -4.266236406216519e-03, -9.116363258336247e-03, -9.116363258336244e-03, -1.278855002579240e-03, -1.278855002579239e-03, -1.209296119776543e-01, -1.209296119776275e-01, -4.912667931254530e-02, -4.912667927285554e-02, -1.229085901056977e+03, -1.229085901056977e+03, -2.172274982004700e+05, -2.172274982004701e+05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
