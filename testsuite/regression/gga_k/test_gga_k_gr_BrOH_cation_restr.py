
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_gr_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_gr", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [1.298951697534464e+03, 1.298930519025128e+03, 1.298884198051001e+03, 1.299198311302010e+03, 1.299026918510820e+03, 1.299026918510820e+03, 4.415250041856439e+01, 4.414552321469777e+01, 4.398648642823723e+01, 4.427990256170911e+01, 4.414626339985082e+01, 4.414626339985082e+01, 2.531729619184472e+00, 2.553281855673697e+00, 3.157469333616098e+00, 3.170635560698011e+00, 3.136518528979027e+00, 3.136518528979027e+00, 6.357178761852433e-01, 6.130271923559452e-01, 3.117448903650777e+00, 7.790641531791875e-01, 6.922472991162741e-01, 6.922472991162745e-01, 7.296296034877904e-01, 7.097317296401984e-01, 9.191176368587975e-01, 7.660812096007602e-01, 6.794392164685165e-01, 6.794392164685157e-01, 6.166357586049373e+01, 6.135477172050859e+01, 6.165100730062530e+01, 6.137836330276341e+01, 6.150609672574909e+01, 6.150609672574909e+01, 3.006979020150458e+01, 2.987281665557432e+01, 3.120643340735757e+01, 3.102951254404718e+01, 2.941704276631667e+01, 2.941704276631667e+01, 1.117328499599336e+00, 5.776713371144329e-01, 1.060826823070718e+00, 5.324917788551510e-01, 1.059168655225444e+00, 1.059168655225444e+00, 9.281682457757040e-01, 1.019807649787273e+00, 9.050001548383050e-01, 5.098019891513381e+00, 7.490023384494026e-01, 7.490023384494026e-01, 6.674755680422375e-01, 7.061459423743923e-01, 2.817236543197165e-01, 8.131223494197987e-01, 4.129834589491074e-01, 4.129834589491079e-01, 3.398795928625088e-01, 4.774360216789307e-01, 4.291219641419732e-01, 3.890689748680296e-01, 4.090980127211332e-01, 4.090980127211332e-01, 2.818667396771609e-01, 1.329169563871325e+00, 1.019388530645805e+00, 7.119646592857896e-01, 8.636792900507495e-01, 8.636792900507495e-01, 6.644019231240648e-01, 1.100471709064700e+00, 1.035429045026316e+00, 7.602759281308684e-01, 8.935351134134748e-01, 8.935351134134746e-01, 1.153274151509030e+00, 9.664666310160918e-01, 9.436831079717253e-01, 5.271305855363119e-01, 7.182623548823993e-01, 7.182623548823990e-01, 8.119190049787266e-01, 3.188784081318976e-01, 4.583053320487870e-01, 7.290045640127011e-01, 3.674732707459731e-01, 3.674732707459726e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_gr_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_gr", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.887786307229243e+02, -3.887523754961654e+02, -3.886819176125567e+02, -3.890715943146568e+02, -3.888608309628934e+02, -3.888608309628934e+02, -2.002686084877362e+01, -2.001977949917226e+01, -1.985715306029291e+01, -2.014764710614551e+01, -2.001933580117388e+01, -2.001933580117388e+01, -1.634185741875885e+00, -1.657716023191814e+00, -2.308843572905064e+00, -2.307863161558590e+00, -2.276764372227707e+00, -2.276764372227707e+00, -5.758759193445619e-01, -5.513589019855647e-01, -1.897082206401255e+00, -7.445739364490210e-01, -6.487568843797360e-01, -6.487568843797358e-01, -7.295570863150420e-01, -7.096513919667782e-01, -9.166722841699966e-01, -7.660570220522107e-01, -6.794010247017608e-01, -6.794010247017596e-01, -7.582415852908861e+00, -7.253752691615567e+00, -7.568858294021497e+00, -7.278683610704465e+00, -7.414910991161664e+00, -7.414910991161664e+00, -2.242888100815031e+01, -2.212065881174015e+01, -2.364623256371400e+01, -2.337155813261274e+01, -2.167540524914539e+01, -2.167540524914539e+01, -4.305591641459463e-01, 2.328270999751607e-01, -4.728258713222944e-01, 1.028919523497665e-01, -3.500763094406419e-01, -3.500763094406419e-01, -9.115428964290195e-01, -9.597837993304298e-01, -8.907712220711764e-01, 2.253458703667082e+00, -7.261062293916170e-01, -7.261062293916170e-01, -6.674611688332416e-01, -7.061228268368873e-01, -2.817101389731337e-01, -8.066856593655062e-01, -4.129638444253696e-01, -4.129638444253697e-01, 3.535751211989212e-01, 2.080858575108703e-01, 2.591861543588320e-01, 3.015383138012584e-01, 2.803579597321935e-01, 2.803579597321935e-01, 3.745031158096104e-01, -8.454915522033957e-01, -4.890145248934359e-01, -1.331500961903670e-01, -3.098688734184820e-01, -3.098688734184820e-01, 2.263979216989657e-01, -1.004657077927025e+00, -8.972378989375287e-01, -5.191774438569648e-01, -7.111984886697140e-01, -7.111984886697135e-01, -7.433188083885859e-01, -9.642264629572077e-01, -9.395193219721152e-01, -3.030717406045385e-01, -7.078263013137174e-01, -7.078263013137169e-01, -8.117745661704809e-01, -3.188767564639857e-01, -4.582980548218699e-01, -7.198020383685038e-01, -3.674564767165787e-01, -3.674564767165786e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_gr_BrOH_cation_restr_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_gr", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [6.465558408889447e-06, 6.465504314368241e-06, 6.465247135945692e-06, 6.466052320295342e-06, 6.465632777057505e-06, 6.465632777057505e-06, 1.498213674029333e-03, 1.498203972832304e-03, 1.497869659939608e-03, 1.497597610737780e-03, 1.498093701227576e-03, 1.498093701227576e-03, 2.087894145121824e-01, 2.094815290984875e-01, 2.271002921854555e-01, 2.215377010814852e-01, 2.227053156391256e-01, 2.227053156391256e-01, 1.212783639246232e+01, 1.159308837356066e+01, 1.316911190273232e-01, 2.771712142089699e+01, 1.957499616309466e+01, 1.957499616309467e+01, 2.874952765754121e+05, 2.465546137483149e+05, 1.468181512983260e+03, 1.492465251230529e+06, 7.522070256195640e+05, 7.522070256195640e+05, 4.463987932311861e-04, 4.461530243974384e-04, 4.463865483604211e-04, 4.461695925050767e-04, 4.462746898771731e-04, 4.462746898771731e-04, 8.405710951977129e-03, 8.225420308312626e-03, 8.540671341115708e-03, 8.377662191483195e-03, 8.242192632024159e-03, 8.242192632024159e-03, 3.119433029115474e-01, 2.433121166868861e-01, 3.937539884428952e-01, 3.505402349909846e-01, 2.973293717959445e-01, 2.973293717959445e-01, 8.282011699658827e+01, 1.207275069070744e+01, 1.046004417152880e+02, 8.906969473199959e-03, 5.124476217261497e+01, 5.124476217261497e+01, 3.249256950004610e+06, 1.597482152034641e+06, 3.573149841128374e+06, 3.437938582102904e+02, 2.043738650702208e+06, 2.043738650702208e+06, 3.074431719955891e-01, 3.127951632558527e-01, 3.108978161438085e-01, 3.093465472861635e-01, 3.101217331306696e-01, 3.101217331306696e-01, 3.338635972182413e-01, 5.277855463435615e-01, 4.596408969843099e-01, 4.031650116453307e-01, 4.307748394546119e-01, 4.307748394546119e-01, 2.111649385648006e-01, 5.986104049931646e+00, 3.455983667980389e+00, 1.499685834607408e+00, 2.280243059795127e+00, 2.280243059795128e+00, 6.763744328316319e-01, 1.674445854271037e+03, 6.607838326576669e+02, 1.673974624492028e+00, 1.665281126261038e+02, 1.665281126261038e+02, 1.022742173599572e+05, 8.363882291868500e+07, 9.043637243187241e+06, 2.011089761174724e+02, 2.579643228134502e+06, 2.579643228134509e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05