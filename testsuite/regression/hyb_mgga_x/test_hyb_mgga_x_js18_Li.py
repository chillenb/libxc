
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_mgga_x_js18_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_js18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-1.765731154796580e+00, -1.242776757958876e+00, -3.612322738776845e-01, -1.703767021229788e-01, -7.655632221344476e-02, -7.594856413304386e-02, -3.283653705300013e-02])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_mgga_x_js18_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_js18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-2.307909120801093e+00, -2.310404736149379e+00, -1.560554954354097e+00, -1.562507019602423e+00, -3.356005109919870e-01, -3.356946525081460e-01, -2.220133294355413e-01, -2.085548586195446e-02, -7.763424053281824e-02, -2.079140800482246e-03, -1.100147673947807e-01, -2.155063733374301e-02, -4.368027799145887e-02, -1.323262558448690e-03])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_js18_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_js18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([-7.704720265345350e-05, 0.000000000000000e+00, -7.494967542301873e-05, -6.861314606715165e-04, 0.000000000000000e+00, -6.753507092652566e-04, -1.099981275811208e-01, 0.000000000000000e+00, -1.106929239530136e-01, -3.198636928973765e+00, 0.000000000000000e+00, -5.044404203727495e+02, -8.199944435229460e+01, 0.000000000000000e+00, -4.240806256537903e+06, 7.184001535704918e+00, 0.000000000000000e+00, -4.453489931123190e+02, 8.451602372304512e+01, 0.000000000000000e+00, -2.576093749850437e+07])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_mgga_x_js18_Li_2_vtau():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_mgga_x_js18", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = numpy.asarray([8.241115651434189e-04, 7.279782203511357e-04, 5.942970122328658e-03, 5.788009787348512e-03, 1.574897416682288e-02, 1.615235375845504e-02, 9.585786965074225e-02, 3.893714147103896e-03, 1.113091268614991e-01, 1.039469295538886e-03, -1.501558390524870e-04, 3.911474973222833e-03, -4.876834205808481e-08, 6.760050798544584e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
