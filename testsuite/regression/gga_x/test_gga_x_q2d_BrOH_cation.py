
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_q2d_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.046815051433614e+01, -2.046818956267087e+01, -2.046840971149108e+01, -2.046782779319322e+01, -2.046812608872686e+01, -2.046812608872686e+01, -3.369265124449230e+00, -3.369245977578145e+00, -3.368885929836237e+00, -3.370151818041691e+00, -3.369323631820613e+00, -3.369323631820613e+00, -6.655346053885399e-01, -6.652367217972419e-01, -6.584654720019232e-01, -6.634166900499443e-01, -6.613878406571665e-01, -6.613878406571665e-01, -1.684882105857144e-01, -1.742212749171651e-01, -7.718575743609355e-01, -6.866928511268966e-02, -1.512373039640925e-01, -1.512373039640924e-01, -4.041634351240318e-04, -4.394386915227852e-04, -4.866693659387262e-03, -1.756558577269260e-04, -2.983481765775111e-04, -2.983481765775113e-04, -4.948171904121033e+00, -4.948186350166492e+00, -4.948180219055164e+00, -4.948192844426199e+00, -4.948174994864011e+00, -4.948174994864011e+00, -1.982744627755550e+00, -1.994526032309999e+00, -1.977996031496229e+00, -1.988510049642007e+00, -1.991434707756365e+00, -1.991434707756365e+00, -5.653716310945007e-01, -5.979391367948814e-01, -5.259600414494734e-01, -5.317587998639045e-01, -5.730751275131640e-01, -5.730751275131640e-01, -1.874117383631548e-02, -1.259145687084034e-01, -1.640446897926872e-02, -1.799722328831302e+00, -3.578551265294896e-02, -3.578551265294896e-02, -1.234376959105515e-04, -1.732252904692234e-04, -1.459887065205442e-04, -9.359789137495442e-03, -1.876999889153270e-04, -1.876999889153270e-04, -5.491913057803572e-01, -5.496330145789289e-01, -5.494833922244027e-01, -5.493505565158567e-01, -5.494168811297913e-01, -5.494168811297913e-01, -5.332569878774285e-01, -4.877905094106277e-01, -5.007673908461445e-01, -5.132588509940184e-01, -5.067633243103417e-01, -5.067633243103417e-01, -6.275117072535690e-01, -2.073802458708370e-01, -2.711170592863088e-01, -3.473314713298608e-01, -3.101249367820554e-01, -3.101249367820554e-01, -4.497277360649197e-01, -4.533273266451437e-03, -6.901046143859063e-03, -3.290383608789864e-01, -1.369933602999111e-02, -1.369933602999111e-02, -6.552636714835833e-04, -2.934395695024826e-05, -8.151602187667041e-05, -1.224537939120745e-02, -1.691124603307051e-04, -1.691124603307049e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_q2d_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.567057138180326e+01, -2.567053994385353e+01, -2.567067460272068e+01, -2.567061491853066e+01, -2.567099406524063e+01, -2.567107569856837e+01, -2.566982864714793e+01, -2.566959018632286e+01, -2.567063348835476e+01, -2.567017606820177e+01, -2.567063348835476e+01, -2.567017606820177e+01, -4.147266370063842e+00, -4.147337513363571e+00, -4.147295322416729e+00, -4.147366592976121e+00, -4.147999851487087e+00, -4.148209126897209e+00, -4.147449514764012e+00, -4.147647802247384e+00, -4.146522581227743e+00, -4.148350534917764e+00, -4.146522581227743e+00, -4.148350534917764e+00, -7.920382661439838e-01, -7.950834922384269e-01, -7.909036775673081e-01, -7.945599042713123e-01, -7.795717708984696e-01, -7.784492700302927e-01, -7.844029234077270e-01, -7.847736843641672e-01, -8.015262445687589e-01, -8.079473374733922e-01, -8.015262445687589e-01, -8.079473374733922e-01, -3.708778743891830e-01, -3.483747883455007e-01, -3.540778918343290e-01, -3.304665575635974e-01, -9.097478627237057e-01, -9.420564086373090e-01, -3.195331230715319e-01, -3.296521084318512e-01, -3.771117603789728e-01, -7.953212266318793e-02, -3.771117603789726e-01, -7.953212266318820e-02, -7.626811869658501e-04, -8.331175024024171e-04, -8.221910810337477e-04, -9.092746289758099e-04, -8.482063396211253e-03, -9.074362636717978e-03, -3.548129455988829e-04, -3.448845919787802e-04, -6.441792087601319e-04, -3.123773562111908e-04, -6.441792087601328e-04, -3.123773562111897e-04, -6.300739072921745e+00, -6.299189425624547e+00, -6.302911050357306e+00, -6.301287664691666e+00, -6.300861600267897e+00, -6.299262863529107e+00, -6.302725255620504e+00, -6.301168995066033e+00, -6.301850069930018e+00, -6.300242984619500e+00, -6.301850069930018e+00, -6.300242984619500e+00, -2.348480637025790e+00, -2.348375428230863e+00, -2.361008671668465e+00, -2.360473797948871e+00, -2.348855769923007e+00, -2.348141059148679e+00, -2.358481724199972e+00, -2.358376599682082e+00, -2.357974666538189e+00, -2.354859359680482e+00, -2.357974666538189e+00, -2.354859359680482e+00, -7.031641185792201e-01, -7.013505756419943e-01, -7.801071874158224e-01, -7.806929876549508e-01, -6.351364236757641e-01, -6.599115135640502e-01, -6.771590846532427e-01, -6.993965961312579e-01, -7.336976505504402e-01, -6.963845575455835e-01, -7.336976505504402e-01, -6.963845575455836e-01, -4.908610650099507e-02, -5.215051993138755e-02, -4.870603178651794e-01, -4.892721735508184e-01, -3.536924085159214e-02, -4.279171364688189e-02, -2.352194926663514e+00, -2.351174420410227e+00, -1.135752547552518e-01, -1.991497726709993e-01, -1.135752547552518e-01, -1.991497726709993e-01, -2.410090159094773e-04, -2.507611925619276e-04, -3.428717281469086e-04, -3.472229771418713e-04, -2.750619307503092e-04, -3.034085630841055e-04, -1.628458654907114e-02, -1.635510190513413e-02, -2.606846270441077e-04, -4.167186922097863e-04, -2.606846270441071e-04, -4.167186922097875e-04, -7.253774596441027e-01, -7.282608904005011e-01, -7.165938619520303e-01, -7.195291327966353e-01, -7.196429422245766e-01, -7.225799326234010e-01, -7.222139329629870e-01, -7.250981494714940e-01, -7.209254032033380e-01, -7.238352338980140e-01, -7.209254032033380e-01, -7.238352338980140e-01, -7.073188686619446e-01, -7.096681571748358e-01, -5.812507862430188e-01, -5.837792572146807e-01, -6.125446944844741e-01, -6.155246521003012e-01, -6.491020115430419e-01, -6.515049739031950e-01, -6.302292005885493e-01, -6.326921908300283e-01, -6.302292005885493e-01, -6.326921908300283e-01, -8.164902735619554e-01, -8.181928327464796e-01, -4.918094555923483e-01, -4.883914441693892e-01, -4.015160323960176e-01, -3.958926291505741e-01, -4.107681160331479e-01, -4.129390093368869e-01, -3.764040277144237e-01, -3.755661302308055e-01, -3.764040277144236e-01, -3.755661302308055e-01, -5.344623081802005e-01, -5.381179323345189e-01, -8.239422985334399e-03, -8.310642210852717e-03, -1.180813196583943e-02, -1.229708719345401e-02, -3.940151811445374e-01, -4.002199011425767e-01, -2.616246388125141e-02, -3.535136002324381e-02, -2.616246388125139e-02, -3.535136002324375e-02, -1.255607519504139e-03, -1.325771449443056e-03, -5.643975559582954e-05, -6.080288890402664e-05, -1.538430195910052e-04, -1.699828413628710e-04, -2.430424972382176e-02, -2.524236032084637e-02, -2.626506342360035e-04, -3.678744792786654e-04, -2.626506342360037e-04, -3.678744792786657e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_q2d_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_q2d", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-8.204721477671555e-09, 0.000000000000000e+00, -8.204766800652357e-09, -8.204653827885079e-09, 0.000000000000000e+00, -8.204717632530637e-09, -8.204365158278215e-09, 0.000000000000000e+00, -8.204324216591065e-09, -8.205130969900808e-09, 0.000000000000000e+00, -8.205299728079567e-09, -8.204684778322386e-09, 0.000000000000000e+00, -8.204856637384994e-09, -8.204684778322386e-09, 0.000000000000000e+00, -8.204856637384994e-09, -1.104263270223588e-05, 0.000000000000000e+00, -1.104632677062645e-05, -1.104288184655165e-05, 0.000000000000000e+00, -1.104685775718459e-05, -1.105175530242992e-05, 0.000000000000000e+00, -1.105381822650998e-05, -1.102922519046162e-05, 0.000000000000000e+00, -1.103201202664016e-05, -1.104880169567010e-05, 0.000000000000000e+00, -1.103896171882926e-05, -1.104880169567010e-05, 0.000000000000000e+00, -1.103896171882926e-05, -6.650929736030813e-03, 0.000000000000000e+00, -6.733387312300981e-03, -6.624842275879034e-03, 0.000000000000000e+00, -6.729797518129655e-03, -6.101594587576283e-03, 0.000000000000000e+00, -5.772534676163467e-03, -5.781900555678419e-03, 0.000000000000000e+00, -5.886400693111606e-03, -6.784832590954974e-03, 0.000000000000000e+00, -2.923979789248508e-03, -6.784832590954974e-03, 0.000000000000000e+00, -2.923979789248508e-03, 2.393041435406563e+00, 0.000000000000000e+00, 1.654846503250253e+00, 2.024390633673546e+00, 0.000000000000000e+00, 1.270196491782043e+00, -3.877787229166219e-03, 0.000000000000000e+00, -3.730700799612941e-03, 6.392401692647081e+00, 0.000000000000000e+00, 6.258130954028600e+00, 1.912370365741008e+00, 0.000000000000000e+00, 3.425864545007339e+00, 1.912370365741006e+00, 0.000000000000000e+00, 3.425864545007357e+00, 8.143857002258817e+01, 0.000000000000000e+00, 7.366373473916418e+01, 7.820490430642955e+01, 0.000000000000000e+00, 7.004472804190674e+01, 2.935013876882243e+00, 0.000000000000000e+00, 2.627359246864987e+00, 1.683767855636213e+02, 0.000000000000000e+00, 1.698173485564390e+02, 9.704468865402897e+01, 0.000000000000000e+00, 4.088904560369648e+02, 9.704468865402970e+01, 0.000000000000000e+00, 4.088904560369617e+02, -2.406518899461401e-06, 0.000000000000000e+00, -2.408754359261360e-06, -2.406482758530346e-06, 0.000000000000000e+00, -2.408720422992840e-06, -2.406499690077036e-06, 0.000000000000000e+00, -2.408740847993340e-06, -2.406469017900232e-06, 0.000000000000000e+00, -2.408710280242898e-06, -2.406514886486870e-06, 0.000000000000000e+00, -2.408740544635677e-06, -2.406514886486870e-06, 0.000000000000000e+00, -2.408740544635677e-06, -6.841183051583497e-05, 0.000000000000000e+00, -6.842036816891723e-05, -6.830269982620767e-05, 0.000000000000000e+00, -6.833333288159917e-05, -6.460353381907075e-05, 0.000000000000000e+00, -6.581947972482105e-05, -6.482249895185300e-05, 0.000000000000000e+00, -6.590503392765956e-05, -7.060438691588929e-05, 0.000000000000000e+00, -6.873780036011448e-05, -7.060438691588929e-05, 0.000000000000000e+00, -6.873780036011448e-05, -1.395359091724174e-02, 0.000000000000000e+00, -1.410930708136091e-02, -1.123741177096296e-02, 0.000000000000000e+00, -1.122833847467967e-02, -1.989711535705817e-02, 0.000000000000000e+00, -1.755388473376415e-02, -1.930663680400419e-02, 0.000000000000000e+00, -1.691686637236593e-02, -1.218694944018629e-02, 0.000000000000000e+00, -1.485116817971022e-02, -1.218694944018629e-02, 0.000000000000000e+00, -1.485116817971022e-02, 1.657863905077823e+00, 0.000000000000000e+00, 1.795417974030410e+00, 2.931129991210610e+00, 0.000000000000000e+00, 2.885078327763911e+00, 1.379594493133019e+00, 0.000000000000000e+00, 1.617382611420636e+00, -1.366751452957456e-04, 0.000000000000000e+00, -1.369281059595690e-04, 3.917174871107204e+00, 0.000000000000000e+00, 7.398084419836229e+00, 3.917174871107204e+00, 0.000000000000000e+00, 7.398084419836229e+00, 3.215720045709976e+02, 0.000000000000000e+00, 2.773248881974216e+02, 2.008102777810884e+02, 0.000000000000000e+00, 1.868015683650553e+02, 9.179157385294604e+02, 0.000000000000000e+00, 8.980138396221271e+02, 1.243555999611279e+00, 0.000000000000000e+00, 1.199948879754921e+00, 5.768626605371516e+02, 0.000000000000000e+00, 3.513519731884950e+02, 5.768626605371513e+02, 0.000000000000000e+00, 3.513519731884969e+02, -1.579481318477810e-02, 0.000000000000000e+00, -1.555767211613558e-02, -1.584608167304243e-02, 0.000000000000000e+00, -1.561097631485389e-02, -1.583247617626980e-02, 0.000000000000000e+00, -1.559645076528515e-02, -1.581821084005022e-02, 0.000000000000000e+00, -1.558175426591401e-02, -1.582583637192475e-02, 0.000000000000000e+00, -1.558954845251050e-02, -1.582583637192475e-02, 0.000000000000000e+00, -1.558954845251050e-02, -1.770435893326536e-02, 0.000000000000000e+00, -1.747286774611165e-02, -2.352999339907651e-02, 0.000000000000000e+00, -2.327999407008195e-02, -2.269035430124602e-02, 0.000000000000000e+00, -2.238930237442985e-02, -2.093158400726304e-02, 0.000000000000000e+00, -2.065881753543039e-02, -2.192506514733582e-02, 0.000000000000000e+00, -2.163026864491728e-02, -2.192506514733582e-02, 0.000000000000000e+00, -2.163026864491728e-02, -9.290462231648465e-03, 0.000000000000000e+00, -9.244767729147566e-03, 9.214579933108207e-01, 0.000000000000000e+00, 8.805686294150891e-01, 1.120058150009308e-01, 0.000000000000000e+00, 8.449866208671296e-02, -8.667032833598026e-02, 0.000000000000000e+00, -8.542947812200188e-02, -7.679929854015932e-02, 0.000000000000000e+00, -7.870092952220099e-02, -7.679929854015918e-02, 0.000000000000000e+00, -7.870092952220108e-02, -3.236597832663116e-02, 0.000000000000000e+00, -3.190703534310029e-02, 2.920946690957220e+00, 0.000000000000000e+00, 2.882323976225162e+00, 1.551412522822916e+00, 0.000000000000000e+00, 1.462532113984674e+00, -1.202953216644470e-01, 0.000000000000000e+00, -1.150166925908201e-01, 1.639311759686405e+00, 0.000000000000000e+00, 2.726537870897319e+00, 1.639311759686404e+00, 0.000000000000000e+00, 2.726537870897311e+00, 4.041739909776098e+01, 0.000000000000000e+00, 3.868979548878060e+01, 3.227145452272716e+03, 0.000000000000000e+00, 4.593629318455018e+03, 8.183450174390265e+02, 0.000000000000000e+00, 7.750401119350214e+02, 1.746389783211880e+00, 0.000000000000000e+00, 1.766290139212266e+00, 9.805184284115089e+02, 0.000000000000000e+00, 4.109279388923225e+02, 9.805184284115193e+02, 0.000000000000000e+00, 4.109279388923250e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05