
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_ow_Li_2_zk():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ow", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = numpy.asarray([-4.580017031229695e-02, -4.333341045520572e-02, -2.782013738938594e-02, -2.279461175667196e-05, -5.763404241921589e-09, -2.226615217334628e-03, -3.212654466848838e-05])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_ow_Li_2_vrho():
    # Prepare the input
    inp = test_data["Li"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ow", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = numpy.asarray([-4.789765462206228e-02, -4.765016267323385e-02, -4.598882939439595e-02, -4.576762577504018e-02, -3.215334649881708e-02, -3.222434832539142e-02, -5.129936801679325e-06, -6.851770914503422e-02, -1.612249465032397e-09, -3.382878319688476e-02, -2.889167613457136e-03, -2.986696585618097e-03, -2.224308559221503e-05, -1.001179648091487e-04])
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05
