
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_b86_mgc_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86_mgc", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.105513663640322e+01, -2.105515857187176e+01, -2.105533625485543e+01, -2.105497546386759e+01, -2.105514753717376e+01, -2.105514753717376e+01, -3.496147118714918e+00, -3.496115198605669e+00, -3.495382330929353e+00, -3.497386806165145e+00, -3.496145594359626e+00, -3.496145594359626e+00, -7.045488008554142e-01, -7.046444176992434e-01, -7.091582641302410e-01, -7.134148303557974e-01, -7.045796302006403e-01, -7.045796302006403e-01, -2.197567744911847e-01, -2.205725996949414e-01, -8.188544393320366e-01, -1.838549197345398e-01, -2.199759663293886e-01, -2.199759663293886e-01, -2.609774798526457e-02, -2.700429868935811e-02, -8.162748569824760e-02, -1.565974925764588e-02, -2.672947203858516e-02, -2.672947203858516e-02, -5.053635182246587e+00, -5.052906207508377e+00, -5.053567269822771e+00, -5.053000335959507e+00, -5.053251782863780e+00, -5.053251782863780e+00, -2.135494479376749e+00, -2.144921162450079e+00, -2.137322935420402e+00, -2.144655824261339e+00, -2.139514345839943e+00, -2.139514345839943e+00, -5.773092322227459e-01, -5.965949494840016e-01, -5.501106495587881e-01, -5.492052967758726e-01, -5.947135096639063e-01, -5.947135096639063e-01, -1.512345988227864e-01, -2.413426251296386e-01, -1.481774771465885e-01, -1.814897316578403e+00, -1.642482206770396e-01, -1.642482206770396e-01, -1.519844154490192e-02, -1.672088086333205e-02, -1.262269599522414e-02, -1.043038505895847e-01, -1.521614283354754e-02, -1.521614283354754e-02, -5.589583380502916e-01, -5.624065302609753e-01, -5.611903842794205e-01, -5.602326201822178e-01, -5.607126387180089e-01, -5.607126387180089e-01, -5.399734587837590e-01, -5.209483064280260e-01, -5.266985194604259e-01, -5.315620709266441e-01, -5.289247248910589e-01, -5.289247248910589e-01, -6.266217174926427e-01, -2.859261732697360e-01, -3.191812118287818e-01, -3.700643235611139e-01, -3.426817215543250e-01, -3.426817215543250e-01, -4.730090238783564e-01, -7.789637925964649e-02, -9.756864132100665e-02, -3.390626336678194e-01, -1.259192077714819e-01, -1.259192077714819e-01, -2.990284591042081e-02, -6.476933479339712e-03, -1.021853964686991e-02, -1.199622533855349e-01, -1.318269895145397e-02, -1.318269895145396e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_b86_mgc_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86_mgc", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.506257937614044e+01, -2.506266610401484e+01, -2.506306037783774e+01, -2.506164004747743e+01, -2.506262505192397e+01, -2.506262505192397e+01, -4.024679577013503e+00, -4.024717206969382e+00, -4.025918019558751e+00, -4.024703046960910e+00, -4.024716075567993e+00, -4.024716075567993e+00, -7.521954169896365e-01, -7.511129130458278e-01, -7.251251329510111e-01, -7.310303756732676e-01, -7.518013544690793e-01, -7.518013544690793e-01, -2.077125649752972e-01, -2.095207699224114e-01, -9.140789633548804e-01, -1.716253202100455e-01, -2.082305540288777e-01, -2.082305540288777e-01, -2.578885229461357e-02, -2.673649802664532e-02, -8.288723859600973e-02, -1.492920478710425e-02, -2.646336550463560e-02, -2.646336550463560e-02, -6.178980203291673e+00, -6.181706023889996e+00, -6.179258630417527e+00, -6.181378097894434e+00, -6.180378043031515e+00, -6.180378043031515e+00, -2.187529775121576e+00, -2.203850361991241e+00, -2.180193744145430e+00, -2.192814222707387e+00, -2.208487872166063e+00, -2.208487872166063e+00, -6.764699228667139e-01, -7.653991430872265e-01, -6.400487765183147e-01, -6.909228668871263e-01, -7.065216634550950e-01, -7.065216634550950e-01, -1.466384077205097e-01, -2.236562728672221e-01, -1.432684579495005e-01, -2.327774790318322e+00, -1.556126003066833e-01, -1.556126003066833e-01, -1.447589817224517e-02, -1.602053882715496e-02, -1.208038752231176e-02, -1.045128462356673e-01, -1.458171196811964e-02, -1.458171196811963e-02, -7.367144408116142e-01, -7.247564453109729e-01, -7.289552977679006e-01, -7.322666944482200e-01, -7.306032414504333e-01, -7.306032414504333e-01, -7.144272011775510e-01, -5.721616633677323e-01, -6.088690906791099e-01, -6.468335058091377e-01, -6.272026153559704e-01, -6.272026153559704e-01, -8.012293263724769e-01, -2.673472270323209e-01, -3.082000610014197e-01, -3.960440083936989e-01, -3.466948651368696e-01, -3.466948651368696e-01, -5.144754306136714e-01, -7.920602488377612e-02, -9.853223908124012e-02, -3.795458042849781e-01, -1.232406211787546e-01, -1.232406211787547e-01, -2.965475908655817e-02, -5.910511694375361e-03, -9.556656464370530e-03, -1.173517910374104e-01, -1.259510271343374e-02, -1.259510271343373e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_b86_mgc_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_b86_mgc", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.623741096952979e-09, -7.623700001330583e-09, -7.623414545756258e-09, -7.624089533823901e-09, -7.623720278165795e-09, -7.623720278165795e-09, -1.019259239626706e-05, -1.019287752617510e-05, -1.019903043729949e-05, -1.017978914187354e-05, -1.019256536908809e-05, -1.019256536908809e-05, -6.367212965778185e-03, -6.366864909791672e-03, -6.267182155380525e-03, -6.116976510735318e-03, -6.367223610503888e-03, -6.367223610503888e-03, -6.640242259003616e-01, -6.582001006431683e-01, -3.433038470581789e-03, -1.105227828281738e+00, -6.626930906744906e-01, -6.626930906744906e-01, -4.021934751570982e+02, -3.608011885617231e+02, -1.034002422389769e+01, -2.015908901339388e+03, -3.756562448731610e+02, -3.756562448731610e+02, -2.265273809139081e-06, -2.265871198019450e-06, -2.265324669324779e-06, -2.265789353434793e-06, -2.265595054753344e-06, -2.265595054753344e-06, -7.619614649703594e-05, -7.482858445946464e-05, -7.598185559067889e-05, -7.492353930602180e-05, -7.552456235202687e-05, -7.552456235202687e-05, -1.359423765752807e-02, -1.135420488707519e-02, -1.654517902767512e-02, -1.598817891770456e-02, -1.198941821217263e-02, -1.198941821217263e-02, -1.758810518298109e+00, -4.143835138578124e-01, -1.956494146343430e+00, -1.325977144797217e-04, -1.531383772029705e+00, -1.531383772029705e+00, -2.249823323672641e+03, -1.635858642804465e+03, -4.976019227025559e+03, -5.362739569088816e+00, -2.390549186784190e+03, -2.390549186784187e+03, -1.450235776488120e-02, -1.434000386117462e-02, -1.439747429897312e-02, -1.444268331942483e-02, -1.442006886661015e-02, -1.442006886661015e-02, -1.661353905520041e-02, -2.109257179543697e-02, -1.974912398014045e-02, -1.855346984189478e-02, -1.918039688874557e-02, -1.918039688874557e-02, -9.347506166415446e-03, -2.267468898414647e-01, -1.519209713722380e-01, -8.359007383426599e-02, -1.150841428359095e-01, -1.150841428359095e-01, -3.114564946194892e-02, -1.121048031866977e+01, -6.042243222673948e+00, -1.166498943934434e-01, -3.354155815202624e+00, -3.354155815202629e+00, -2.458475244756054e+02, -4.535702110092952e+04, -9.167213550160102e+03, -4.091599485584485e+00, -4.107963802296457e+03, -4.107963802296475e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05