
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_hpbeint_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.714105467057565e+01, -1.714108654385889e+01, -1.714126824756302e+01, -1.714079321906052e+01, -1.714103643552270e+01, -1.714103643552270e+01, -2.863739560203354e+00, -2.863725311352690e+00, -2.863465426798234e+00, -2.864452994619186e+00, -2.863790614044807e+00, -2.863790614044807e+00, -5.914154596917984e-01, -5.910547064990912e-01, -5.838250667434033e-01, -5.880118310955222e-01, -5.890801740611172e-01, -5.890801740611172e-01, -1.794880479284709e-01, -1.808703720666498e-01, -6.823977724402647e-01, -1.503868623836109e-01, -1.711243301644313e-01, -1.711243301644312e-01, -8.421613003887782e-03, -8.867914201556667e-03, -4.813300953616476e-02, -4.857566935810692e-03, -6.778829623827407e-03, -6.778829623827407e-03, -4.191893101932092e+00, -4.192058048367084e+00, -4.191906323780354e+00, -4.192051817992338e+00, -4.191973371373033e+00, -4.191973371373033e+00, -1.708872113139506e+00, -1.717889749009545e+00, -1.707355996186202e+00, -1.715265383185723e+00, -1.714677177411415e+00, -1.714677177411415e+00, -5.148632090713720e-01, -5.562289697633045e-01, -4.795889114894539e-01, -4.967393320422951e-01, -5.227476933820768e-01, -5.227476933820768e-01, -1.154808689918439e-01, -1.901762306207923e-01, -1.080881510936127e-01, -1.575482521480268e+00, -1.284554414220528e-01, -1.284554414220528e-01, -3.750409539332256e-03, -4.748533689882492e-03, -3.636143370240675e-03, -7.567427219684118e-02, -4.567449875437552e-03, -4.567449875437557e-03, -5.188681415054350e-01, -5.148457651718042e-01, -5.161935616381123e-01, -5.173588928992990e-01, -5.167694911331137e-01, -5.167694911331137e-01, -5.063738670489041e-01, -4.410994547577141e-01, -4.572021749565646e-01, -4.750282837477404e-01, -4.656528248746424e-01, -4.656528248746424e-01, -5.809947842858294e-01, -2.271175579776846e-01, -2.586727719037146e-01, -3.198235298237243e-01, -2.858152999065846e-01, -2.858152999065846e-01, -4.084579974881742e-01, -4.610004028873535e-02, -6.210950933735992e-02, -3.075305322070510e-01, -9.313653363255553e-02, -9.313653363255553e-02, -1.186549484015654e-02, -1.269388974332833e-03, -2.669356728440974e-03, -8.825477353917878e-02, -4.194543332233718e-03, -4.194543332233710e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_hpbeint_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.141866750889873e+01, -2.141864222756247e+01, -2.141876179806324e+01, -2.141871106909019e+01, -2.141904421967466e+01, -2.141911929657821e+01, -2.141797521587349e+01, -2.141776193761922e+01, -2.141872060110760e+01, -2.141829283423884e+01, -2.141872060110760e+01, -2.141829283423884e+01, -3.537320426434770e+00, -3.537475902917282e+00, -3.537361306408017e+00, -3.537523142849571e+00, -3.538413955557454e+00, -3.538675390304597e+00, -3.537145627727282e+00, -3.537412793946760e+00, -3.536698467593667e+00, -3.538368403420014e+00, -3.536698467593667e+00, -3.538368403420014e+00, -7.218847920884104e-01, -7.263997964625813e-01, -7.199202208509277e-01, -7.254558592519514e-01, -6.895424586715286e-01, -6.820785323085485e-01, -6.907521981384227e-01, -6.931782398975548e-01, -7.264511143976452e-01, -6.541879669277570e-01, -7.264511143976452e-01, -6.541879669277570e-01, -1.863569729136914e-01, -1.900214114043176e-01, -1.896130073788016e-01, -1.941708069721452e-01, -8.241122745621562e-01, -8.562141327964549e-01, -1.496866890814781e-01, -1.506234980011908e-01, -1.716902665823982e-01, -1.448177407034584e-01, -1.716902665823981e-01, -1.448177407034583e-01, -1.084590968995826e-02, -1.152258325256589e-02, -1.136272432684807e-02, -1.216888632623948e-02, -6.051955142175824e-02, -6.354018054645751e-02, -6.527153062763938e-03, -6.418569679025541e-03, -9.678987746662614e-03, -5.511642133912077e-03, -9.678987746662614e-03, -5.511642133912077e-03, -5.371341924338979e+00, -5.370080013779930e+00, -5.373532106543849e+00, -5.372202248535158e+00, -5.371461709727124e+00, -5.370154751309974e+00, -5.373345529839660e+00, -5.372078097761959e+00, -5.372463095746460e+00, -5.371147291701511e+00, -5.372463095746460e+00, -5.371147291701511e+00, -1.902936398104119e+00, -1.902839132812120e+00, -1.920647216625088e+00, -1.920087608747512e+00, -1.880861084090259e+00, -1.886433874763013e+00, -1.896180015417146e+00, -1.901881333660880e+00, -1.928683213341705e+00, -1.915172163276095e+00, -1.928683213341705e+00, -1.915172163276095e+00, -6.665072925179792e-01, -6.653128920165914e-01, -7.319127761021260e-01, -7.323631617346373e-01, -6.082850199564319e-01, -6.260835158401108e-01, -6.458939184175179e-01, -6.605032207323999e-01, -6.907120210170768e-01, -6.650005771493112e-01, -6.907120210170768e-01, -6.650005771493112e-01, -1.282173858777391e-01, -1.282644095837961e-01, -1.867626434651561e-01, -1.871359772914127e-01, -1.205199902712927e-01, -1.234952386779354e-01, -2.076531267325417e+00, -2.075756950218704e+00, -1.354512378888050e-01, -1.307573948678787e-01, -1.354512378888050e-01, -1.307573948678787e-01, -4.897578808254003e-03, -5.089761114239225e-03, -6.280204364396548e-03, -6.375293613444572e-03, -4.695214120259203e-03, -4.971655602338162e-03, -9.186809214474799e-02, -9.251387431876694e-02, -4.796480827542009e-03, -6.583668719713746e-03, -4.796480827542018e-03, -6.583668719713751e-03, -6.798820128600451e-01, -6.817988743163836e-01, -6.764299712692862e-01, -6.784084662569285e-01, -6.778431190210465e-01, -6.798148254133504e-01, -6.788681376607537e-01, -6.807924979434441e-01, -6.783748703195496e-01, -6.803221181046265e-01, -6.783748703195496e-01, -6.803221181046265e-01, -6.624303010291240e-01, -6.639711400868322e-01, -5.481706848735128e-01, -5.502544816009511e-01, -5.860644531564155e-01, -5.883120762007666e-01, -6.211027767367124e-01, -6.227446428992629e-01, -6.039673157112810e-01, -6.056775250085905e-01, -6.039673157112810e-01, -6.056775250085905e-01, -7.641949954517604e-01, -7.654420918555682e-01, -2.289070038157607e-01, -2.296270378449256e-01, -2.794884476514180e-01, -2.816367936205720e-01, -3.975196854339513e-01, -3.991799931037597e-01, -3.352702658035104e-01, -3.353762391835726e-01, -3.352702658035102e-01, -3.353762391835726e-01, -5.074798932620791e-01, -5.106140320397969e-01, -5.959106349630126e-02, -5.996038950141489e-02, -7.751756072497602e-02, -7.957913207626366e-02, -3.923975395347504e-01, -3.967919728870274e-01, -1.063173581889007e-01, -1.065626724169012e-01, -1.063173581889007e-01, -1.065626724169012e-01, -1.549244937526503e-02, -1.604978048902046e-02, -1.690494003289238e-03, -1.694310094789802e-03, -3.439118071460568e-03, -3.657613625966953e-03, -1.021408984481813e-01, -1.032202621292517e-01, -4.542500695522871e-03, -6.035391684949340e-03, -4.542500695522859e-03, -6.035391684949331e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_hpbeint_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_hpbeint", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-7.369859469483371e-09, 3.910730011692272e-10, -7.369898870049557e-09, -7.369763814390597e-09, 3.910796170225820e-10, -7.369828471254866e-09, -7.369437629284019e-09, 3.910880515325018e-10, -7.369370328521318e-09, -7.370530218710012e-09, 3.909900735927581e-10, -7.370749277653227e-09, -7.369820038878639e-09, 3.910443870235806e-10, -7.370170940015530e-09, -7.369820038878639e-09, 3.910443870235806e-10, -7.370170940015530e-09, -9.584067818136503e-06, 2.169598145258871e-06, -9.585166544644788e-06, -9.583792952094360e-06, 2.170010452975426e-06, -9.585007045285224e-06, -9.578634837091194e-06, 2.179313971358895e-06, -9.578251928855463e-06, -9.581688102222791e-06, 2.161176390734945e-06, -9.581659388485850e-06, -9.590883343868151e-06, 2.169837283211506e-06, -9.576036098079823e-06, -9.590883343868151e-06, 2.169837283211506e-06, -9.576036098079823e-06, -5.733142586818628e-03, 3.558615888277719e-03, -5.651479927108392e-03, -5.777463833793050e-03, 3.537505481054945e-03, -5.677408488246818e-03, -6.499915208834688e-03, 2.985740826017254e-03, -6.627036288009926e-03, -6.372970079435175e-03, 2.912978377606038e-03, -6.335471453544265e-03, -5.779440313774265e-03, 2.953093828184752e-03, -6.840093101048497e-03, -5.779440313774265e-03, 2.953093828184752e-03, -6.840093101048497e-03, -9.572806879486864e-01, 3.755362477810255e-01, -8.631898431300640e-01, -9.394189390813771e-01, 3.905720907018519e-01, -8.275972433412188e-01, -3.369246425876406e-03, 1.912538072893896e-03, -2.896958206215079e-03, -1.527167481828012e+00, 3.137833963629834e-01, -1.472060059391695e+00, -8.541458515208216e-01, 2.541200338556246e-01, -2.273263403816903e+00, -8.541458515208211e-01, 2.541200338556253e-01, -2.273263403816905e+00, -4.083748517080723e+00, 1.394083716767340e-02, -4.043527477110812e+00, -4.302833990480708e+00, 1.639695140223212e-02, -4.281874367146900e+00, -2.435773141118978e+00, 8.343743235418925e-02, -2.459517542022418e+00, -3.740603935069136e+00, 4.952961555121697e-03, -3.638902497258025e+00, -4.078934454001598e+00, 7.266611993229210e-03, -1.034877964244913e+01, -4.078934454002421e+00, 7.266611991447444e-03, -1.034877964244997e+01, -1.875796822267144e-06, 5.536058940772718e-07, -1.877852316531313e-06, -1.872673684809925e-06, 5.570421809896916e-07, -1.874775594376027e-06, -1.875640818128581e-06, 5.537428597096498e-07, -1.877730366070363e-06, -1.872918759392699e-06, 5.567761054803036e-07, -1.874979640350361e-06, -1.874210532324318e-06, 5.553540316891062e-07, -1.876297211796985e-06, -1.874210532324318e-06, 5.553540316891062e-07, -1.876297211796985e-06, -9.173378286883155e-05, 1.337308237690807e-05, -9.175266158833551e-05, -8.928516592401484e-05, 1.329660849359038e-05, -8.938122302986418e-05, -9.342680405116005e-05, 1.281776404362160e-05, -9.305513091848788e-05, -9.129132364443278e-05, 1.275550308861711e-05, -9.088625091824749e-05, -8.912229396238902e-05, 1.360706952123280e-05, -9.021858495676640e-05, -8.912229396238902e-05, 1.360706952123280e-05, -9.021858495676640e-05, -7.790333453010551e-03, 1.074167850091081e-02, -7.931720034636210e-03, -3.452684262870437e-03, 1.238170522618870e-02, -3.435890463606987e-03, -1.227611640111056e-02, 1.409341863113520e-02, -9.732860317187121e-03, -7.170961690653010e-03, 1.915267401784013e-02, -5.116551978521958e-03, -5.974556612962768e-03, 1.062684737733898e-02, -8.520925780061177e-03, -5.974556612962768e-03, 1.062684737733898e-02, -8.520925780061176e-03, -1.800448261328329e+00, 1.931328258527610e-01, -1.834660508916256e+00, -6.721928844014549e-01, 1.521045390677060e-01, -6.641655048426642e-01, -2.020620709669498e+00, 1.974971516713018e-01, -1.970793971101260e+00, -6.380688997003535e-05, 1.064376481511526e-04, -6.401784569391211e-05, -1.921495738204328e+00, 3.510009466248905e-01, -2.313478062889468e+00, -1.921495738204328e+00, 3.510009466248905e-01, -2.313478062889468e+00, -5.251219731890830e+00, 4.829525739717181e-03, -4.544797881123626e+00, -4.521629374215465e+00, 6.089571670993928e-03, -4.174459016597657e+00, -2.574747255246151e+01, 6.065915863432825e-02, -2.864968035877319e+01, -2.913818577327510e+00, 1.991525682306227e-01, -2.802652841807708e+00, -1.280470304908990e+01, 2.321577969903203e-02, -1.258324798425567e+01, -1.280470304909014e+01, 2.321577969244254e-02, -1.258324798425594e+01, -3.245824718452700e-03, 2.006658329285461e-02, -3.042583253347723e-03, -4.668137366405473e-03, 1.781467759800113e-02, -4.458754883831729e-03, -4.200094954675836e-03, 1.854372342028741e-02, -3.992214634296547e-03, -3.786945277005172e-03, 1.919701127442328e-02, -3.582081363934944e-03, -3.996509327112997e-03, 1.886455508012585e-02, -3.790150172727788e-03, -3.996509327112997e-03, 1.886455508012585e-02, -3.790150172727788e-03, -3.072440496482991e-03, 2.346972384260296e-02, -2.878263461698706e-03, -1.828502892052182e-02, 1.544340176380472e-02, -1.786885482659171e-02, -1.373016054522336e-02, 1.696281722981208e-02, -1.335688811242820e-02, -9.417618298537756e-03, 1.897612528914740e-02, -9.156393600874498e-03, -1.156003784834943e-02, 1.793593116057567e-02, -1.126205119815729e-02, -1.156003784834943e-02, 1.793593116057567e-02, -1.126205119815729e-02, -3.047529512138786e-03, 9.904618307524754e-03, -2.996571300339029e-03, -3.557638957508121e-01, 1.057743991314675e-01, -3.502304047389741e-01, -2.071229545436161e-01, 8.924935641299944e-02, -2.008363693393577e-01, -6.951911528801749e-02, 6.757769337377416e-02, -6.739275323984126e-02, -1.238370749784884e-01, 8.163707956642091e-02, -1.243939837838192e-01, -1.238370749784885e-01, 8.163707956642095e-02, -1.243939837838192e-01, -2.520984591172312e-02, 2.213917364013108e-02, -2.430852966083029e-02, -2.225833941859686e+00, 6.866400901121213e-02, -2.229510961252210e+00, -2.226603221851506e+00, 1.043435010147752e-01, -2.284784888472339e+00, -7.466997115042379e-02, 1.014865553695685e-01, -6.770134801363001e-02, -2.921453640194482e+00, 3.380561560592040e-01, -3.490069506256027e+00, -2.921453640194483e+00, 3.380561560592035e-01, -3.490069506256026e+00, -3.237265758118712e+00, 1.737567135607801e-02, -3.305775096598428e+00, -1.615652978500331e+01, 6.604821135914521e-03, -2.862210886746537e+01, -9.971144463227084e+00, 8.265978807604456e-03, -1.061565315661570e+01, -3.284923104555799e+00, 3.126730447295191e-01, -3.217861421099719e+00, -2.641274089122893e+01, 2.963288318679947e-02, -1.305834863541181e+01, -2.641274089123072e+01, 2.963288318747713e-02, -1.305834863541359e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05