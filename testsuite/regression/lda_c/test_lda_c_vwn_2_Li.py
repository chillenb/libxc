
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_vwn_2_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-9.400974897587693e-02, -8.420806801095740e-02, -4.968925324814365e-02, -1.805906217327232e-02, -1.097300102564883e-02, -6.794977311230624e-03, -1.437883917436271e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_vwn_2_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_vwn_2", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.031844365902482e-01, -1.030201000379599e-01, -9.308633451061274e-02, -9.294564977139695e-02, -5.684145948068858e-02, -5.688458985417118e-02, -2.097186000744139e-02, -1.272540209227145e-01, -1.310880057968793e-02, -7.296187747771522e-02, -8.539145127109756e-03, -8.647747442968595e-03, -1.280651323149347e-04, -3.692740008748051e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
