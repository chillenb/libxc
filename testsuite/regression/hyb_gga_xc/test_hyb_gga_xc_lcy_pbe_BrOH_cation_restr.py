
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_lcy_pbe_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.060009253865632e+01, -2.060012165297934e+01, -2.060031607642836e+01, -2.059988157781890e+01, -2.060010000263021e+01, -2.060010000263021e+01, -3.142808953709116e+00, -3.142783104344788e+00, -3.142270148151647e+00, -3.143815592841693e+00, -3.142860869517368e+00, -3.142860869517368e+00, -4.354773818518917e-01, -4.349260796133904e-01, -4.223291395856692e-01, -4.271223753472767e-01, -4.259396230109860e-01, -4.259396230109860e-01, -4.203223404441894e-02, -4.375282492621624e-02, -5.359140272141401e-01, -2.083056939085333e-02, -2.834212431551618e-02, -2.834212431551618e-02, -2.432155877759993e-06, -2.836741554558174e-06, -4.734189134279218e-04, -4.680558148692418e-07, -9.292148654327456e-07, -9.292148654327456e-07, -4.726895703912303e+00, -4.726649554741917e+00, -4.726893323970974e+00, -4.726675867085330e+00, -4.726765805806716e+00, -4.726765805806716e+00, -1.746258473697175e+00, -1.757549078213678e+00, -1.743637679336909e+00, -1.753620607393503e+00, -1.753762432041581e+00, -1.753762432041581e+00, -3.604231832036863e-01, -4.080672389899361e-01, -3.230786242078169e-01, -3.458443647864787e-01, -3.687691865847015e-01, -3.687691865847015e-01, -7.686393279658508e-03, -3.997044810287913e-02, -6.195984572306052e-03, -1.565213833711096e+00, -1.219532712608913e-02, -1.219532712608913e-02, -2.149821656663470e-07, -4.373385613071591e-07, -1.960891046474254e-07, -1.994435999853339e-03, -3.423496482591120e-07, -3.423496482591120e-07, -3.708809568946684e-01, -3.653736253797520e-01, -3.672209804304205e-01, -3.688170224826309e-01, -3.680101261506582e-01, -3.680101261506582e-01, -3.586022042772897e-01, -2.795049718786343e-01, -3.000564635971584e-01, -3.217764980753119e-01, -3.105129985780782e-01, -3.105129985780782e-01, -4.338580108367533e-01, -6.691445926380153e-02, -9.736424467071339e-02, -1.637122271492738e-01, -1.272713031885833e-01, -1.272713031885833e-01, -2.471290672666040e-01, -4.148346504243822e-04, -1.041730951007925e-03, -1.573140084193899e-01, -4.073483809522901e-03, -4.073483809522748e-03, -6.838279082275710e-06, -8.350681644010483e-09, -7.725482423460745e-08, -3.390773907441711e-03, -2.713051819558700e-07, -2.713051819580377e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_lcy_pbe_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.494881506958826e+01, -2.494891471521333e+01, -2.494933198072934e+01, -2.494784999534870e+01, -2.494863017901378e+01, -2.494863017901378e+01, -3.854706068822460e+00, -3.854750093367411e+00, -3.855867432414386e+00, -3.854690787637216e+00, -3.854855947645042e+00, -3.854855947645042e+00, -6.124461940003928e-01, -6.112986494381532e-01, -5.835912602103666e-01, -5.897869422817439e-01, -5.887788821082420e-01, -5.887788821082420e-01, -8.647231509899053e-02, -9.001900883053995e-02, -7.380508685376698e-01, -4.375318002534787e-02, -5.970630936171806e-02, -5.970630936171804e-02, -4.887744408584193e-06, -5.704825075563120e-06, -9.753425380548595e-04, -9.378038320315544e-07, -1.864675992221123e-06, -1.864675992218780e-06, -6.009142659970762e+00, -6.011857467013514e+00, -6.009264410295242e+00, -6.011661076244209e+00, -6.010520254750522e+00, -6.010520254750522e+00, -2.017173234480919e+00, -2.034418687202336e+00, -2.002620880221460e+00, -2.017676317633037e+00, -2.033957900798876e+00, -2.033957900798876e+00, -5.335506165678642e-01, -5.965992523977557e-01, -4.835217208391169e-01, -5.126397742420090e-01, -5.453614933357696e-01, -5.453614933357696e-01, -1.606307767827725e-02, -7.713445513815489e-02, -1.301593619549213e-02, -2.137702461496131e+00, -2.627915973933713e-02, -2.627915973933713e-02, -4.306353797872554e-07, -8.764802765613758e-07, -3.954964015356281e-07, -4.228824597971369e-03, -6.883716636294793e-07, -6.883716636321787e-07, -5.388792109565594e-01, -5.372304956599913e-01, -5.380705740937287e-01, -5.385621490487802e-01, -5.383402830481469e-01, -5.383402830481469e-01, -5.191692363344976e-01, -4.212496763663183e-01, -4.527429751747539e-01, -4.827627118002976e-01, -4.677679538151976e-01, -4.677679538151976e-01, -6.325495859330826e-01, -1.217761972752279e-01, -1.704006939412893e-01, -2.692978295944486e-01, -2.166666859435658e-01, -2.166666859435658e-01, -3.795925499172079e-01, -8.508541789799371e-04, -2.157887067678334e-03, -2.604102285750236e-01, -8.883315923009833e-03, -8.883315923009828e-03, -1.376541632867608e-05, -1.671888397857695e-08, -1.547991998958697e-07, -7.360207186814309e-03, -5.458443919143975e-07, -5.458443919150882e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_lcy_pbe_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_lcy_pbe", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-6.659625738099329e-09, -6.659575157562417e-09, -6.659291364721969e-09, -6.660045104843316e-09, -6.659658537724198e-09, -6.659658537724198e-09, -6.954662249272738e-06, -6.954693548974569e-06, -6.954772314381908e-06, -6.949763360484133e-06, -6.954084124470087e-06, -6.954084124470087e-06, -9.014651786915428e-04, -9.135459727499392e-04, -1.180939914252070e-03, -1.159661102520890e-03, -1.157065409591952e-03, -1.157065409591952e-03, 1.210692562829169e-01, 1.281970863239217e-01, -7.113178064209908e-04, 9.309808401704817e-02, 1.177215835648695e-01, 1.177215835648691e-01, 4.259488913977694e-03, 5.018496372487451e-03, 2.552714612757480e-02, 1.504930900101684e-03, 3.145657694110334e-03, 3.145657692713386e-03, -1.571261558812273e-06, -1.569647586219618e-06, -1.571186260961094e-06, -1.569761828809263e-06, -1.570447182717818e-06, -1.570447182717818e-06, -4.800840124849228e-05, -4.710831598381969e-05, -4.814055907192890e-05, -4.735450585953187e-05, -4.743015895572963e-05, -4.743015895572963e-05, 9.831332333879693e-04, 3.103302231835423e-03, 1.568086110650115e-03, 5.104865626970446e-03, 1.134278232662139e-03, 1.134278232662139e-03, 5.656804622398262e-02, 3.705530745423362e-02, 5.882370688716106e-02, -2.902713098572557e-05, 1.071239448446481e-01, 1.071239448446481e-01, 1.469903487914531e-03, 1.850146306491007e-03, 1.851300569674611e-02, 6.070345949222385e-02, 8.056791862840744e-03, 8.056791871762886e-03, 6.774601819080920e-03, 5.077227685430918e-03, 5.613045615525057e-03, 6.104026591938813e-03, 5.852818969830676e-03, 5.852818969830676e-03, 8.493241137194768e-03, 7.681194215570326e-04, 2.144700273744711e-03, 4.002318797788515e-03, 3.010717511069806e-03, 3.010717511069806e-03, 2.211172896904429e-03, 2.170786024251982e-02, 1.795012444550868e-02, 1.644823770277977e-02, 1.819770917703588e-02, 1.819770917703591e-02, 2.305877567017636e-03, 2.087719234242101e-02, 3.171937382236794e-02, 3.216703312101745e-02, 1.045419159664819e-01, 1.045419159664856e-01, 5.290823591179625e-03, 2.005778828866134e-03, 2.523867611089677e-03, 9.607946789562330e-02, 1.003261621755490e-02, 1.003261621571892e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05