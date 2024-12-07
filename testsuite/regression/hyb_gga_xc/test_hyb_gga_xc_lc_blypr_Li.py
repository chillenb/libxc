
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lc_blypr_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_blypr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.665414254613071e+00, -1.141548581546742e+00, -1.806478933441737e-01, -4.361529873019205e-02, -4.022858367609117e-03, -2.649097659949182e-05, -1.818420961946826e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lc_blypr_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_blypr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.137276476595852e+00, -2.139202565137261e+00, -1.431586730565630e+00, -1.432758550017543e+00, -3.043563466877997e-01, -3.044142599594176e-01, -7.561311771903564e-02, -4.328542226754114e-02, -7.749010987553829e-03, -9.048414874070699e-03, -5.359558903325027e-05, -5.243736644434678e-05, -4.378326221243755e-10, -1.573075221436466e-10]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lc_blypr_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lc_blypr", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.247567723220023e-04, 6.377456738086418e-06, -2.241354023324337e-04, -7.621867076135584e-04, 4.413025517428745e-05, -7.603822295472548e-04, -7.938564490899892e-03, 3.253606485805580e-02, -7.848149715712398e-03, -4.413017382531119e-01, 7.216492012456489e-02, 5.406107601049400e-02, -2.548403466192122e-01, 2.566359735238707e-08, 6.959826531149451e-09, -6.427967498472996e-05, 2.763910643695453e-153, -6.015208008458597e-05, -3.058001984680710e-09, 0.000000000000000e+00, -1.479347944172945e-09]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
