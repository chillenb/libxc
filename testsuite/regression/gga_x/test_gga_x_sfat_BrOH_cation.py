
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_x_sfat_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sfat", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.082385410495670e+01, -2.082388143275061e+01, -2.082407142007113e+01, -2.082366342584388e+01, -2.082386746945442e+01, -2.082386746945442e+01, -3.264027824906467e+00, -3.263993846813381e+00, -3.263293570972725e+00, -3.265171805712003e+00, -3.264071200635329e+00, -3.264071200635329e+00, -5.039056683560128e-01, -5.036578401561642e-01, -4.985958565813147e-01, -5.034077179975274e-01, -5.033240244324605e-01, -5.033240244324605e-01, -6.698209789799008e-02, -6.859157308885182e-02, -6.090033870970095e-01, -4.029970244711374e-02, -6.349888478268671e-02, -6.349888478268670e-02, -7.111554831357739e-06, -8.311172118900279e-06, -1.346729758507145e-03, -1.359875771430018e-06, -3.975194327777474e-06, -3.975194327777474e-06, -4.836765549613617e+00, -4.836171787259246e+00, -4.836748917895052e+00, -4.836224579477700e+00, -4.836458771752538e+00, -4.836458771752538e+00, -1.879297090785012e+00, -1.889941051353839e+00, -1.878657419182598e+00, -1.888062827936623e+00, -1.885504185372352e+00, -1.885504185372352e+00, -4.052463133294122e-01, -4.342473271913837e-01, -3.677160762821422e-01, -3.722207534310313e-01, -4.125029550411840e-01, -4.125029550411841e-01, -1.815708624265432e-02, -7.030362581070221e-02, -1.506132085528746e-02, -1.618118806683121e+00, -2.602674502067050e-02, -2.602674502067050e-02, -6.263239205794480e-07, -1.270319431468843e-06, -5.718374992345143e-07, -5.288725117205418e-03, -1.187104638739620e-06, -1.187104638739621e-06, -3.872718967164442e-01, -3.885766950691261e-01, -3.881582700211716e-01, -3.877763614929770e-01, -3.879705154251264e-01, -3.879705154251264e-01, -3.720920746414583e-01, -3.312424333567474e-01, -3.438153948640558e-01, -3.554433197882444e-01, -3.494890372385889e-01, -3.494890372385889e-01, -4.625261534957213e-01, -1.054770180129478e-01, -1.391478637227700e-01, -2.011653141015190e-01, -1.677956967720602e-01, -1.677956967720602e-01, -2.952847208597897e-01, -1.178093341160869e-03, -2.891626924869953e-03, -1.860533039172838e-01, -1.006925147853117e-02, -1.006925147853117e-02, -1.987482269321061e-05, -2.425218413837925e-08, -2.261941979184466e-07, -8.518568826049979e-03, -9.135894559333934e-07, -9.135894559333911e-07]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_x_sfat_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sfat", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.502202750747541e+01, -2.502199725312270e+01, -2.502213664868325e+01, -2.502207652435969e+01, -2.502246485196959e+01, -2.502255268508263e+01, -2.502123283678700e+01, -2.502098220681341e+01, -2.502209366005049e+01, -2.502159454855373e+01, -2.502209366005049e+01, -2.502159454855373e+01, -3.881828578229181e+00, -3.881940354159917e+00, -3.881862162224468e+00, -3.881976739684124e+00, -3.882705654071948e+00, -3.882950639936691e+00, -3.881898326825719e+00, -3.882138780419153e+00, -3.881085218388878e+00, -3.882962561498514e+00, -3.881085218388878e+00, -3.882962561498514e+00, -6.188357895214996e-01, -6.224230336298836e-01, -6.174697221916169e-01, -6.218569906111459e-01, -5.997641230839100e-01, -5.941462367940725e-01, -6.021473477634747e-01, -6.039414285574475e-01, -6.292030859344625e-01, -5.754140234019886e-01, -6.292030859344625e-01, -5.754140234019886e-01, -9.188193465345287e-02, -9.750331784397319e-02, -9.350957383592504e-02, -1.000495018641649e-01, -7.306050563796516e-01, -7.633273320666816e-01, -5.995520301230494e-02, -6.146591395044564e-02, -9.876486992971290e-02, -3.277685351689657e-02, -9.876486992971287e-02, -3.277685351689659e-02, -1.282244044225581e-05, -1.538378164917781e-05, -1.475276147406300e-05, -1.813411532954861e-05, -2.409299096663872e-03, -2.815352210235581e-03, -2.786359205421682e-06, -2.649396627984476e-06, -9.104665836273744e-06, -1.678277726522756e-06, -9.104665836273744e-06, -1.678277726522757e-06, -6.014770596975082e+00, -6.013233377078326e+00, -6.017209784538668e+00, -6.015589696908259e+00, -6.014906565951184e+00, -6.013314685477566e+00, -6.016999492933983e+00, -6.015455265096233e+00, -6.016019328397203e+00, -6.014416559013440e+00, -6.016019328397203e+00, -6.014416559013440e+00, -2.053924730726536e+00, -2.053809985495985e+00, -2.071520710206690e+00, -2.070884087747881e+00, -2.036777927471309e+00, -2.041579880296993e+00, -2.051915832932754e+00, -2.056911246362925e+00, -2.076733776185792e+00, -2.064550244540524e+00, -2.076733776185792e+00, -2.064550244540524e+00, -5.344459551988499e-01, -5.327138871827621e-01, -6.029629535748140e-01, -6.034926904986243e-01, -4.713909640385694e-01, -4.941524237703196e-01, -5.055497608072957e-01, -5.266624792486612e-01, -5.625513848352732e-01, -5.274879563956707e-01, -5.625513848352732e-01, -5.274879563956708e-01, -3.014740268825566e-02, -3.072694716335689e-02, -9.771519931224466e-02, -9.847865080180894e-02, -2.415841153951404e-02, -2.711579483428793e-02, -2.139477761181353e+00, -2.138483034679025e+00, -3.934215601793919e-02, -4.341444410629686e-02, -3.934215601793919e-02, -4.341444410629686e-02, -1.176459855996707e-06, -1.320460783606161e-06, -2.482014929842335e-06, -2.596417773512980e-06, -1.037843728200426e-06, -1.232699610783535e-06, -9.732427215116884e-03, -9.896161467362949e-03, -1.105746214041099e-06, -2.863725534223958e-06, -1.105746214041098e-06, -2.863725534223959e-06, -5.503191832185874e-01, -5.530699142163750e-01, -5.423790849062782e-01, -5.451471027830961e-01, -5.450534817638797e-01, -5.478334652561513e-01, -5.473704341535061e-01, -5.501114401688985e-01, -5.462012906763912e-01, -5.489613649406078e-01, -5.462012906763912e-01, -5.489613649406078e-01, -5.331369104212665e-01, -5.353774141852624e-01, -4.219448798882424e-01, -4.243600934455576e-01, -4.504390652825715e-01, -4.531275687139991e-01, -4.813016502486188e-01, -4.835198635389183e-01, -4.653864422645948e-01, -4.676499597029241e-01, -4.653864422645948e-01, -4.676499597029241e-01, -6.378726964633953e-01, -6.394567662100882e-01, -1.393019550303062e-01, -1.404519432420982e-01, -1.797639369892192e-01, -1.822197174788379e-01, -2.651330066506858e-01, -2.671757219349106e-01, -2.191570449655433e-01, -2.190246123596412e-01, -2.191570449655433e-01, -2.190246123596413e-01, -3.784843467776893e-01, -3.819902856388691e-01, -2.283716836646645e-03, -2.328896048801366e-03, -5.268646755835457e-03, -5.764250647325215e-03, -2.507586145675667e-01, -2.561727179351362e-01, -1.664136819863764e-02, -1.878739436107586e-02, -1.664136819863754e-02, -1.878739436107579e-02, -3.750079755414671e-05, -4.172341388931650e-05, -4.833762710618166e-08, -4.866979876970986e-08, -4.072129145567421e-07, -4.899263333258615e-07, -1.496849617094960e-02, -1.557383624066135e-02, -9.397184320282346e-07, -2.205198800081080e-06, -9.397184320282328e-07, -2.205198800081074e-06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_x_sfat_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_x_sfat", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.428443591606255e-08, 0.000000000000000e+00, -1.428451751295198e-08, -1.428435649225861e-08, 0.000000000000000e+00, -1.428445975267971e-08, -1.428392651746801e-08, 0.000000000000000e+00, -1.428389287671067e-08, -1.428482799474098e-08, 0.000000000000000e+00, -1.428504047878411e-08, -1.428439742794947e-08, 0.000000000000000e+00, -1.428445233319647e-08, -1.428439742794947e-08, 0.000000000000000e+00, -1.428445233319647e-08, -1.734033088049422e-05, 0.000000000000000e+00, -1.734690435023783e-05, -1.734082671116725e-05, 0.000000000000000e+00, -1.734789132097579e-05, -1.735782406298301e-05, 0.000000000000000e+00, -1.736189256554357e-05, -1.731747238772123e-05, 0.000000000000000e+00, -1.732272771450839e-05, -1.734942088547731e-05, 0.000000000000000e+00, -1.733633058056904e-05, -1.734942088547731e-05, 0.000000000000000e+00, -1.733633058056904e-05, -7.296769953963004e-03, 0.000000000000000e+00, -7.325365325218231e-03, -7.292354865124030e-03, 0.000000000000000e+00, -7.327093741750087e-03, -7.248959785049166e-03, 0.000000000000000e+00, -7.208470011755154e-03, -7.050125063498818e-03, 0.000000000000000e+00, -7.066018398134875e-03, -7.324141760880041e-03, 0.000000000000000e+00, -6.800834787067976e-03, -7.324141760880041e-03, 0.000000000000000e+00, -6.800834787067976e-03, -2.428995353280845e-01, 0.000000000000000e+00, -2.369316307614231e-01, -2.453135463584396e-01, 0.000000000000000e+00, -2.375782287363682e-01, -4.493902571905251e-03, 0.000000000000000e+00, -4.274653526461956e-03, -2.545210954860033e-01, 0.000000000000000e+00, -2.515484490847443e-01, -2.230702225886784e-01, 0.000000000000000e+00, -2.415069188089554e-01, -2.230702225886783e-01, 0.000000000000000e+00, -2.415069188089586e-01, -5.948168367809707e-04, 0.000000000000000e+00, -6.966971351274863e-04, -6.997786375578455e-04, 0.000000000000000e+00, -8.402926057546096e-04, -4.025580660504390e-02, 0.000000000000000e+00, -4.589235906914366e-02, -1.386328449738912e-04, 0.000000000000000e+00, -1.298391971517791e-04, -4.353025794074037e-04, 0.000000000000000e+00, -1.754894557359699e-04, -4.353025794074401e-04, 0.000000000000000e+00, -1.754894557371710e-04, -4.092620739645475e-06, 0.000000000000000e+00, -4.096233712918625e-06, -4.095678785862572e-06, 0.000000000000000e+00, -4.099192348797106e-06, -4.092747487706900e-06, 0.000000000000000e+00, -4.096304526740571e-06, -4.095371783115349e-06, 0.000000000000000e+00, -4.098992574048885e-06, -4.094219198289770e-06, 0.000000000000000e+00, -4.097724242910410e-06, -4.094219198289770e-06, 0.000000000000000e+00, -4.097724242910410e-06, -1.217516968716904e-04, 0.000000000000000e+00, -1.217724082035184e-04, -1.193975606389295e-04, 0.000000000000000e+00, -1.194977581116105e-04, -1.218668606262474e-04, 0.000000000000000e+00, -1.218803569795693e-04, -1.198372368287551e-04, 0.000000000000000e+00, -1.198004231694154e-04, -1.202925189569278e-04, 0.000000000000000e+00, -1.205664577805788e-04, -1.202925189569278e-04, 0.000000000000000e+00, -1.205664577805788e-04, -1.410702364280655e-02, 0.000000000000000e+00, -1.424806666894234e-02, -1.334335834051685e-02, 0.000000000000000e+00, -1.336853167515343e-02, -1.889131160706362e-02, 0.000000000000000e+00, -1.708335566543499e-02, -2.086728145527474e-02, 0.000000000000000e+00, -1.854828275011754e-02, -1.270375830033006e-02, 0.000000000000000e+00, -1.502961185276518e-02, -1.270375830033007e-02, 0.000000000000000e+00, -1.502961185276518e-02, -2.008607552505538e-01, 0.000000000000000e+00, -2.049881303248515e-01, -1.744983280950827e-01, 0.000000000000000e+00, -1.736315729636394e-01, -1.905118978155661e-01, 0.000000000000000e+00, -2.001173353867616e-01, -2.253345502566017e-04, 0.000000000000000e+00, -2.257634987207907e-04, -2.429140332418667e-01, 0.000000000000000e+00, -2.841517075054133e-01, -2.429140332418667e-01, 0.000000000000000e+00, -2.841517075054133e-01, -7.902945989574610e-05, 0.000000000000000e+00, -7.955597460939078e-05, -1.420698415990406e-04, 0.000000000000000e+00, -1.401431602078568e-04, -2.115872310621896e-04, 0.000000000000000e+00, -2.665798377517234e-04, -1.315863536328778e-01, 0.000000000000000e+00, -1.301447952343610e-01, -1.384958143900873e-04, 0.000000000000000e+00, -3.276696603019390e-04, -1.384958143905361e-04, 0.000000000000000e+00, -3.276696603019448e-04, -1.938199557753195e-02, 0.000000000000000e+00, -1.915484937850290e-02, -1.813714962278272e-02, 0.000000000000000e+00, -1.793237071060827e-02, -1.850876984065417e-02, 0.000000000000000e+00, -1.829971294607687e-02, -1.886663059909463e-02, 0.000000000000000e+00, -1.864579651158902e-02, -1.868171332530600e-02, 0.000000000000000e+00, -1.846684390174685e-02, -1.868171332530600e-02, 0.000000000000000e+00, -1.846684390174685e-02, -2.205937998568134e-02, 0.000000000000000e+00, -2.181299557840358e-02, -2.166740204087331e-02, 0.000000000000000e+00, -2.144848216978953e-02, -2.107963073993595e-02, 0.000000000000000e+00, -2.086773117846770e-02, -2.088286147944367e-02, 0.000000000000000e+00, -2.066872022547454e-02, -2.097000552199022e-02, 0.000000000000000e+00, -2.074476330591716e-02, -2.097000552199022e-02, 0.000000000000000e+00, -2.074476330591716e-02, -1.117180295059010e-02, 0.000000000000000e+00, -1.116527373160714e-02, -1.266532011131604e-01, 0.000000000000000e+00, -1.257878497118031e-01, -9.832823640836919e-02, 0.000000000000000e+00, -9.730937084173347e-02, -6.435725864089852e-02, 0.000000000000000e+00, -6.350806684066834e-02, -8.068047094562224e-02, 0.000000000000000e+00, -8.107328538874942e-02, -8.068047094562231e-02, 0.000000000000000e+00, -8.107328538874947e-02, -2.841649001279649e-02, 0.000000000000000e+00, -2.801349355988743e-02, -3.639693414115107e-02, 0.000000000000000e+00, -3.701681301092799e-02, -7.073945166245811e-02, 0.000000000000000e+00, -7.691618479701223e-02, -8.193740232560746e-02, 0.000000000000000e+00, -7.923145996842877e-02, -1.911332823728357e-01, 0.000000000000000e+00, -2.282652333490606e-01, -1.911332823728319e-01, 0.000000000000000e+00, -2.282652333490593e-01, -1.337342119998183e-03, 0.000000000000000e+00, -1.491990516875327e-03, -8.785116538737534e-06, 0.000000000000000e+00, -1.323703547764535e-05, -4.610387836069480e-05, 0.000000000000000e+00, -5.720515159415277e-05, -1.906028413566972e-01, 0.000000000000000e+00, -1.932154723578976e-01, -1.964863937751822e-04, 0.000000000000000e+00, -2.647209976134371e-04, -1.964863937761352e-04, 0.000000000000000e+00, -2.647209976129070e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05