
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_xc_hle16_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hle16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.638072303825458e+01, -2.638080096599196e+01, -2.638116286046434e+01, -2.638000315838397e+01, -2.638060861511040e+01, -2.638060861511040e+01, -4.326842246721061e+00, -4.326848016283956e+00, -4.327087995373521e+00, -4.327487747496139e+00, -4.326953038555454e+00, -4.326953038555454e+00, -8.636681567426129e-01, -8.631678211651964e-01, -8.546448865171918e-01, -8.606591325372513e-01, -8.669856440185627e-01, -8.669856440185627e-01, -2.816468219991783e-01, -2.813088381864423e-01, -1.000518795447290e+00, -2.625767129335313e-01, -2.706297024709085e-01, -2.706297024709085e-01, -1.510620922562088e-02, -1.598926817237061e-02, -9.851479633962218e-02, -8.393274123363364e-03, -1.662506253081576e-02, -1.662506253081576e-02, -6.475264707758351e+00, -6.476842728976853e+00, -6.475339852140913e+00, -6.476732818772343e+00, -6.476062628598152e+00, -6.476062628598152e+00, -2.522637170055799e+00, -2.536237861100717e+00, -2.520138357013515e+00, -2.531870490631533e+00, -2.531752863495974e+00, -2.531752863495974e+00, -7.480827234547039e-01, -8.148808691109882e-01, -6.945895363259674e-01, -7.218035322890989e-01, -7.603908809990891e-01, -7.603908809990894e-01, -2.247720952880877e-01, -3.191348660214006e-01, -2.125469164723283e-01, -2.420465856869729e+00, -2.346334725053995e-01, -2.346334725053995e-01, -6.421387702471489e-03, -8.194892061276497e-03, -6.241539021132084e-03, -1.539015244491274e-01, -9.267486379565516e-03, -9.267486379565522e-03, -7.603053184795554e-01, -7.508051656220440e-01, -7.537373925026137e-01, -7.564495179040277e-01, -7.550554108018828e-01, -7.550554108018828e-01, -7.428643057709071e-01, -6.382782748824368e-01, -6.607085853254324e-01, -6.874338917519989e-01, -6.731657694124656e-01, -6.731657694124656e-01, -8.530262459437752e-01, -3.605521770117583e-01, -3.902542020638411e-01, -4.585856227388297e-01, -4.168027774514030e-01, -4.168027774514030e-01, -5.896131241256957e-01, -9.404017351080375e-02, -1.280696150303666e-01, -4.367861853228449e-01, -1.826418699643593e-01, -1.826418699643592e-01, -2.170153831057047e-02, -2.096483745292858e-03, -4.535931470780054e-03, -1.751209895570051e-01, -8.273226136294387e-03, -8.273226136294385e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_xc_hle16_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hle16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.556130845816725e+01, -3.556126624661975e+01, -3.556146564632062e+01, -3.556138027814080e+01, -3.556193406783552e+01, -3.556206172004021e+01, -3.556016146797339e+01, -3.555980007274210e+01, -3.556140615365747e+01, -3.556067725478247e+01, -3.556140615365747e+01, -3.556067725478247e+01, -5.711644497129297e+00, -5.712076434338925e+00, -5.711721925412387e+00, -5.712175235881630e+00, -5.713818966482758e+00, -5.714423309313289e+00, -5.711100345075266e+00, -5.711740813186367e+00, -5.710500488867446e+00, -5.713704869155960e+00, -5.710500488867446e+00, -5.713704869155960e+00, -1.049459154776030e+00, -1.062077734840250e+00, -1.044960372478834e+00, -1.060512548631949e+00, -9.831812693042685e-01, -9.592109023585564e-01, -9.782963182339168e-01, -9.861289398857722e-01, -1.083692632786341e+00, -8.455892326693815e-01, -1.083692632786341e+00, -8.455892326693815e-01, -1.925415561472349e-01, -2.013339719880254e-01, -1.969147379936814e-01, -2.084194751815139e-01, -1.208554538720075e+00, -1.283328819881142e+00, -1.660820395793737e-01, -1.630411582006157e-01, -1.734599880386683e-01, -3.826819572880821e-01, -1.734599880386681e-01, -3.826819572880823e-01, -1.730479327175903e-02, -2.301922167917543e-02, -1.788638007500437e-02, -2.464526017951450e-02, -1.172490563060075e-01, -1.282637617874161e-01, -1.179464135788449e-02, -1.082470350343370e-02, -2.781637853591692e-02, 8.757933573111739e-03, -2.781637853591690e-02, 8.757933573111271e-03, -8.754364632500335e+00, -8.752230620522319e+00, -8.757279602381155e+00, -8.755044014685145e+00, -8.754531002287358e+00, -8.752328635604226e+00, -8.757029844157909e+00, -8.754887343664560e+00, -8.755858034910657e+00, -8.753644392440101e+00, -8.755858034910657e+00, -8.753644392440101e+00, -2.834916889276943e+00, -2.834733815296075e+00, -2.872237070400817e+00, -2.871088716655606e+00, -2.772684345304343e+00, -2.790962606420232e+00, -2.805586194558712e+00, -2.823768783045500e+00, -2.900587301783470e+00, -2.860093897562776e+00, -2.900587301783470e+00, -2.860093897562776e+00, -9.810671918873158e-01, -9.787788798828976e-01, -1.097460784707929e+00, -1.098320224027671e+00, -8.822687559534528e-01, -9.186680752930055e-01, -9.551177157847353e-01, -9.845490602468872e-01, -1.025473441687153e+00, -9.746482169147522e-01, -1.025473441687153e+00, -9.746482169147526e-01, -1.876940362034376e-01, -1.802611388285357e-01, -1.918662611700400e-01, -1.913911686022323e-01, -1.934223197525528e-01, -1.720438458317739e-01, -3.267179221350283e+00, -3.265769489961456e+00, -1.920721526770788e-01, -1.404406948753744e-01, -1.920721526770788e-01, -1.404406948753744e-01, -7.719269243396374e-03, -9.475496117442125e-03, -1.061594038647144e-02, -1.146982847177977e-02, -7.034397292426668e-03, -9.547798419409287e-03, -1.616534752439360e-01, -1.641026232252518e-01, -8.033350883626163e-04, -1.687634151228562e-02, -8.033350883623581e-04, -1.687634151228548e-02, -1.018566859837455e+00, -1.022329395929863e+00, -1.009469383100734e+00, -1.013425936632219e+00, -1.013173774655932e+00, -1.017066470120821e+00, -1.015785231790652e+00, -1.019567576656616e+00, -1.014534972440270e+00, -1.018367467365924e+00, -1.014534972440270e+00, -1.018367467365924e+00, -9.925004497638040e-01, -9.955760665672185e-01, -7.788991818166314e-01, -7.835144134340379e-01, -8.493134462574502e-01, -8.540921351318185e-01, -9.118053241602553e-01, -9.151644667974396e-01, -8.812565153083142e-01, -8.847455655034511e-01, -8.812565153083142e-01, -8.847455655034511e-01, -1.147964194858956e+00, -1.150393034598753e+00, -2.375093372519828e-01, -2.388991582175194e-01, -3.177971699071144e-01, -3.244699084205934e-01, -5.401469952247326e-01, -5.438635967277493e-01, -4.242503763367184e-01, -4.252335489719951e-01, -4.242503763367187e-01, -4.252335489719957e-01, -7.137900372518821e-01, -7.210053878910233e-01, -1.185053435032788e-01, -1.199999477838429e-01, -1.495561092933637e-01, -1.518757772624168e-01, -5.403822661291685e-01, -5.503306073110135e-01, -1.852876769984367e-01, -1.447876453210065e-01, -1.852876769984365e-01, -1.447876453210058e-01, -2.700770957049097e-02, -3.148752647701406e-02, -2.791326486283891e-03, -2.827780796242573e-03, -4.981736286229581e-03, -7.022633880074590e-03, -1.658517201648881e-01, -1.626298198632379e-01, -1.603382078560633e-03, -1.512257567083643e-02, -1.603382078560774e-03, -1.512257567083638e-02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_xc_hle16_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_xc_hle16", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [2.046938599076179e-09, 0.000000000000000e+00, 2.046968641467127e-09, 2.047161846430967e-09, 0.000000000000000e+00, 2.047129723942507e-09, 2.047534232374105e-09, 0.000000000000000e+00, 2.047763880474979e-09, 2.045034050407099e-09, 0.000000000000000e+00, 2.044560141944304e-09, 2.047100277231566e-09, 0.000000000000000e+00, 2.045582622445936e-09, 2.047100277231566e-09, 0.000000000000000e+00, 2.045582622445936e-09, -1.471283010628969e-06, 0.000000000000000e+00, -1.456367354915093e-06, -1.469481688967515e-06, 0.000000000000000e+00, -1.453623756318927e-06, -1.416162384023816e-06, 0.000000000000000e+00, -1.401486846815871e-06, -1.507464318421041e-06, 0.000000000000000e+00, -1.490858056542725e-06, -1.478659439639722e-06, 0.000000000000000e+00, -1.443265188012824e-06, -1.478659439639722e-06, 0.000000000000000e+00, -1.443265188012824e-06, -6.613534502343948e-03, 0.000000000000000e+00, -6.066944967814398e-03, -6.796556877174750e-03, 0.000000000000000e+00, -6.125240796053313e-03, -9.181066332170322e-03, 0.000000000000000e+00, -1.012989901977596e-02, -9.379246433461243e-03, 0.000000000000000e+00, -9.079014572227991e-03, -4.996291262138933e-03, 0.000000000000000e+00, -1.405334703028638e-02, -4.996291262138933e-03, 0.000000000000000e+00, -1.405334703028638e-02, -2.769358079470789e+00, 0.000000000000000e+00, -2.445428658204076e+00, -2.688041483424533e+00, 0.000000000000000e+00, -2.312721893199718e+00, -3.653322229589201e-03, 0.000000000000000e+00, -2.248566230626698e-03, -5.064194093461888e+00, 0.000000000000000e+00, -4.974221288885209e+00, -2.540325656760784e+00, 0.000000000000000e+00, 6.707921797572818e+00, -2.540325656760787e+00, 0.000000000000000e+00, 6.707921797572907e+00, 2.249074749607681e+00, 0.000000000000000e+00, -2.292983760375573e+01, 4.362724349111650e+00, 0.000000000000000e+00, -2.605175114910769e+01, -8.600089780671095e+00, 0.000000000000000e+00, -1.749296073595815e+01, -1.231433129785355e+01, 0.000000000000000e+00, -3.887127537201993e+00, -5.764191727906355e+01, 0.000000000000000e+00, 4.948980969421290e+02, -5.764191727896404e+01, 0.000000000000000e+00, 4.948980969445791e+02, 1.057861482238329e-06, 0.000000000000000e+00, 1.058145021868357e-06, 1.071716359232556e-06, 0.000000000000000e+00, 1.071523612448352e-06, 1.058568793962777e-06, 0.000000000000000e+00, 1.058548288799277e-06, 1.070440561400788e-06, 0.000000000000000e+00, 1.070720047669676e-06, 1.065013582841020e-06, 0.000000000000000e+00, 1.064867436588315e-06, 1.065013582841020e-06, 0.000000000000000e+00, 1.064867436588315e-06, -1.198324785500267e-04, 0.000000000000000e+00, -1.198637704155744e-04, -1.139470560775530e-04, 0.000000000000000e+00, -1.141317985665463e-04, -1.293628280676504e-04, 0.000000000000000e+00, -1.265453982618893e-04, -1.240083700607373e-04, 0.000000000000000e+00, -1.212425756739754e-04, -1.096105072525889e-04, 0.000000000000000e+00, -1.159292359386357e-04, -1.096105072525889e-04, 0.000000000000000e+00, -1.159292359386357e-04, -3.283658671595374e-03, 0.000000000000000e+00, -3.305131977680642e-03, 1.051237190298555e-02, 0.000000000000000e+00, 1.087607965284010e-02, -8.422915179748544e-03, 0.000000000000000e+00, -5.397613382480038e-03, 1.201652820070828e-02, 0.000000000000000e+00, 1.052512582413622e-02, -1.235628669350724e-03, 0.000000000000000e+00, -2.688335621930577e-03, -1.235628669350681e-03, 0.000000000000000e+00, -2.688335621930564e-03, -7.792213264157914e+00, 0.000000000000000e+00, -8.264553119651463e+00, -2.116233561123008e+00, 0.000000000000000e+00, -2.094460413448826e+00, -8.031959385442565e+00, 0.000000000000000e+00, -9.928320249621290e+00, 1.329051312843258e-04, 0.000000000000000e+00, 1.332798502772378e-04, -6.610999146616751e+00, 0.000000000000000e+00, -8.723737499695881e+00, -6.610999146616751e+00, 0.000000000000000e+00, -8.723737499695881e+00, -4.101486686326859e+00, 0.000000000000000e+00, -1.395425101474502e+01, -8.650701304495470e+00, 0.000000000000000e+00, -1.030446387149719e+01, 6.407945013232943e+01, 0.000000000000000e+00, -1.566198176050126e+02, -1.496814871422009e+01, 0.000000000000000e+00, -1.467945483443067e+01, 3.389191094900814e+02, 0.000000000000000e+00, -1.574765027543597e+02, 3.389191094875349e+02, 0.000000000000000e+00, -1.574765027545014e+02, 3.315391709145098e-02, 0.000000000000000e+00, 3.332743401017522e-02, 1.572669656494807e-02, 0.000000000000000e+00, 1.590512984024745e-02, 1.991865312553318e-02, 0.000000000000000e+00, 2.007664790679682e-02, 2.438825263028314e-02, 0.000000000000000e+00, 2.443815807990872e-02, 2.200262588578490e-02, 0.000000000000000e+00, 2.210536049630502e-02, 2.200262588578490e-02, 0.000000000000000e+00, 2.210536049630502e-02, 6.052572844426179e-02, 0.000000000000000e+00, 6.004834739258022e-02, -2.246963876209207e-02, 0.000000000000000e+00, -2.168808539136787e-02, -1.035151580645787e-02, 0.000000000000000e+00, -9.711731174794149e-03, 8.790999737165891e-04, 0.000000000000000e+00, 1.110946199393597e-03, -4.824225359523742e-03, 0.000000000000000e+00, -4.532126554286205e-03, -4.824225359523742e-03, 0.000000000000000e+00, -4.532126554286205e-03, 7.749927919308550e-03, 0.000000000000000e+00, 8.146082876453272e-03, -1.016912394307509e+00, 0.000000000000000e+00, -9.984807437457212e-01, -5.244413248394242e-01, 0.000000000000000e+00, -5.014607391937059e-01, -1.101356378510698e-01, 0.000000000000000e+00, -1.064248732276997e-01, -2.616935142631596e-01, 0.000000000000000e+00, -2.608817625195125e-01, -2.616935142631595e-01, 0.000000000000000e+00, -2.608817625195128e-01, -3.299826533718963e-02, 0.000000000000000e+00, -3.109167358097777e-02, -1.172656199523899e+01, 0.000000000000000e+00, -1.278254059359746e+01, -1.020468335348923e+01, 0.000000000000000e+00, -1.477642280880525e+01, -9.774556300456411e-02, 0.000000000000000e+00, -8.621352338658005e-02, -1.034367881669502e+01, 0.000000000000000e+00, -1.799236621836081e+01, -1.034367881669520e+01, 0.000000000000000e+00, -1.799236621836088e+01, -4.334301210658659e+00, 0.000000000000000e+00, -1.701598573267687e+01, 5.541603574099430e+01, 0.000000000000000e+00, -1.493088864387248e+02, 2.741658703265934e+01, 0.000000000000000e+00, -5.548412267939666e+01, -1.464176141438344e+01, 0.000000000000000e+00, -1.566304030991046e+01, 3.474258645915571e+02, 0.000000000000000e+00, -1.291220767158966e+02, 3.474258645927196e+02, 0.000000000000000e+00, -1.291220767158948e+02]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05