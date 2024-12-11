
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_c_acgga_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_acgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-6.155181882468815e-02, -4.534982135558897e-02, -3.798094949056738e-03, -1.523930555047293e-02, -1.503265634489524e-03, -1.331092027399838e-08, -3.691166296226900e-16])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_c_acgga_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_acgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-1.150256867650166e-01, -1.148983776346154e-01, -1.009950301840160e-01, -1.008985278419865e-01, -1.851454410466881e-02, -1.852125178772706e-02, -2.407309625157115e-02, -9.772839125403270e-02, -7.037075050878860e-03, 3.850288188804456e-01, -8.380096004960747e-08, -8.422391368786604e-08, -2.303990602903899e-15, -2.726964975444807e-15])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_c_acgga_Li_2_vsigma():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_c_acgga", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = numpy.asarray([4.344359952306926e-05, 8.688719904613852e-05, 4.344359952306926e-05, 1.417858156358072e-04, 2.835716312716144e-04, 1.417858156358072e-04, 3.557094290958459e-03, 7.114188581916917e-03, 3.557094290958459e-03, 3.257806215255809e+00, 6.515612430511617e+00, 3.257806215255809e+00, 1.187442098453823e+01, 2.374884196907647e+01, 1.187442098453823e+01, 2.825065736786347e-04, 5.650131473572694e-04, 2.825065736786347e-04, 3.243028016144399e-06, 6.486056032288799e-06, 3.243028016144399e-06])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
