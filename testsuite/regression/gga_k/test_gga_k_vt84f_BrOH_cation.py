
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_gga_k_vt84f_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vt84f", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [2.909417205016653e+03, 2.909400963813629e+03, 2.909385855137329e+03, 2.909626321204020e+03, 2.909492203345299e+03, 2.909492203345299e+03, 8.803359772713510e+01, 8.802616316700320e+01, 8.785858071085970e+01, 8.818337945634401e+01, 8.802893393364059e+01, 8.802893393364059e+01, 4.134902959885787e+00, 4.148513765686543e+00, 4.493887585036688e+00, 4.540673484908438e+00, 4.530168675113292e+00, 4.530168675113292e+00, 6.602467741121891e-01, 6.411477664374275e-01, 5.337339191587537e+00, 7.795104884202585e-01, 7.190758649816623e-01, 7.190758649816623e-01, 7.296064448879009e-01, 7.097030059527608e-01, 9.182601892335036e-01, 7.661093275562577e-01, 6.863130727509509e-01, 6.863130727509501e-01, 1.552971136627938e+02, 1.549869229637468e+02, 1.552847683291774e+02, 1.550108855748217e+02, 1.551386918675197e+02, 1.551386918675197e+02, 4.161114845973559e+01, 4.177451058194322e+01, 4.217780133648272e+01, 4.232264956775069e+01, 4.142734265279776e+01, 4.142734265279776e+01, 2.351286569167709e+00, 1.982740406593554e+00, 2.130655655924778e+00, 1.626665552492348e+00, 2.325361353610755e+00, 2.325361353610755e+00, 9.245274877354170e-01, 1.027966526305780e+00, 9.016211763939089e-01, 1.787038503799278e+01, 7.492945044617504e-01, 7.492945044617504e-01, 6.677868763604320e-01, 7.065331033207147e-01, 2.817690296425823e-01, 8.112145369420909e-01, 4.137919138135489e-01, 4.137919138135494e-01, 1.597873365268500e+00, 1.668056471232930e+00, 1.637651775551300e+00, 1.616840519982164e+00, 1.626698403388700e+00, 1.626698403388700e+00, 1.498452403348975e+00, 2.198394956563611e+00, 1.987993503745887e+00, 1.720338030265280e+00, 1.853824306220037e+00, 1.853824306220037e+00, 2.203843835451725e+00, 1.134446972916096e+00, 1.129261885604125e+00, 1.175711044228327e+00, 1.114935874563437e+00, 1.114935874563437e+00, 1.886225846021984e+00, 9.656728947525863e-01, 9.423758146490735e-01, 9.382069495618101e-01, 7.170549414668186e-01, 7.170549414668185e-01, 8.120358554787405e-01, 3.205942872776386e-01, 4.583345272537480e-01, 7.266203712855601e-01, 3.710531471965939e-01, 3.710531471965934e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_gga_k_vt84f_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vt84f", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [1.931869266237501e+03, 1.931867171038641e+03, 1.931917854628540e+03, 1.931902447613180e+03, 1.932027483833373e+03, 1.932072371968713e+03, 1.931480023196518e+03, 1.931374235606554e+03, 1.931900558949739e+03, 1.931619545845791e+03, 1.931900558949739e+03, 1.931619545845791e+03, 4.465479930560303e+01, 4.466609221468462e+01, 4.465663706226552e+01, 4.466857858669754e+01, 4.470853528596178e+01, 4.472300611082358e+01, 4.463429095810584e+01, 4.464889223764528e+01, 4.463497743587033e+01, 4.469552636396362e+01, 4.463497743587033e+01, 4.469552636396362e+01, 1.980678349828219e+00, 1.923969801974954e+00, 1.998973315462123e+00, 1.929778210446463e+00, 2.138244678029726e+00, 2.102405458460700e+00, 2.159830513098218e+00, 2.167167730573829e+00, 1.849808726870491e+00, 8.012323512721322e-01, 1.849808726870491e+00, 8.012323512721322e-01, -4.568814558418735e-01, -4.112293923498154e-01, -4.222687218542743e-01, -3.685702477245004e-01, 2.563398708220394e+00, 2.403177910133256e+00, -7.068728180476770e-01, -7.111360467011220e-01, -4.756261506018173e-01, -7.701063372248528e-01, -4.756261506018172e-01, -7.701063372248526e-01, -7.276658706104248e-01, -7.312231602286198e-01, -7.087820017888717e-01, -7.104514170642502e-01, -9.213548301117774e-01, -9.150736624083486e-01, -7.609624726482568e-01, -7.715222141049977e-01, -7.284688027584519e-01, -4.575543790839710e-01, -7.284688027584508e-01, -4.575543790839705e-01, 1.365122464551355e+02, 1.364238354675295e+02, 1.371319892981849e+02, 1.370223539573947e+02, 1.365443667564030e+02, 1.364427604472263e+02, 1.370761260772200e+02, 1.369864767085598e+02, 1.368311020236603e+02, 1.367244378458858e+02, 1.368311020236603e+02, 1.367244378458858e+02, 1.892745582329476e+01, 1.892506824471634e+01, 1.935191098748803e+01, 1.933870333423510e+01, 1.799031087250050e+01, 1.831335274698095e+01, 1.851699637167594e+01, 1.879288448912245e+01, 1.950448103740063e+01, 1.921061274824952e+01, 1.950448103740063e+01, 1.921061274824952e+01, 1.339344124857538e+00, 1.333903912733393e+00, 2.702255491445481e+00, 2.720433811601815e+00, 1.043154378879825e+00, 1.146335644001515e+00, 1.910772181536253e+00, 2.022479587013170e+00, 1.548876308074748e+00, 1.365518817364464e+00, 1.548876308074746e+00, 1.365518817364465e+00, -9.178822688190252e-01, -9.038370515157207e-01, -8.705715646770696e-01, -8.687738572130433e-01, -8.939987658941421e-01, -8.887915114254068e-01, 2.487470173835405e+01, 2.485960344723685e+01, -7.907985564289015e-01, -6.531508107432912e-01, -7.907985564289015e-01, -6.531508107432912e-01, -6.423526105376139e-01, -6.904473126496586e-01, -6.921222180719154e-01, -7.203089107386033e-01, -2.899527964655255e-01, -2.748789309454749e-01, -8.010042990290600e-01, -8.165899929124258e-01, -4.112409563495210e-01, -4.147767267054233e-01, -4.112409563495210e-01, -4.147767267054236e-01, 2.561295123771238e+00, 2.583853644472667e+00, 2.301666449549504e+00, 2.328101704233136e+00, 2.404159295159843e+00, 2.429985150656459e+00, 2.481940515637350e+00, 2.505467268715134e+00, 2.444199718990027e+00, 2.468818605631219e+00, 2.444199718990027e+00, 2.468818605631219e+00, 2.463220004716211e+00, 2.479746087528770e+00, 1.030876603140513e+00, 1.030360920213012e+00, 9.673274355783974e-01, 9.776571885345167e-01, 1.369354594210956e+00, 1.385689533239002e+00, 1.099308311771989e+00, 1.111817266779344e+00, 1.099308311771989e+00, 1.111817266779344e+00, 2.903357893676332e+00, 2.940160157716884e+00, -7.967279438031621e-01, -7.884989572303835e-01, -4.279309753785961e-01, -3.782254251817648e-01, 5.682505345028964e-01, 5.707989524417765e-01, 2.995821597864259e-01, 3.088039376946921e-01, 2.995821597864254e-01, 3.088039376946923e-01, 8.915157129773349e-01, 8.882219039955647e-01, -9.659399014709479e-01, -9.649389659606137e-01, -9.490715461004133e-01, -9.346931847159419e-01, 4.270379207741135e-01, 4.317028675265648e-01, -7.550975083776078e-01, -6.715681582091522e-01, -7.550975083776079e-01, -6.715681582091527e-01, -8.165368042995080e-01, -8.079887325491747e-01, -3.662903374316486e-01, -2.752101338691900e-01, -4.661839751434474e-01, -4.518103103038786e-01, -7.187893319700744e-01, -7.236114525621975e-01, -2.863697929975417e-01, -4.071377321336060e-01, -2.863697929975412e-01, -4.071377321336056e-01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_gga_k_vt84f_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("gga_k_vt84f", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [1.477154585875062e-05, 0.000000000000000e+00, 1.477160353799303e-05, 1.477141039085423e-05, 0.000000000000000e+00, 1.477150512044184e-05, 1.477093842961254e-05, 0.000000000000000e+00, 1.477083960636424e-05, 1.477246885587000e-05, 0.000000000000000e+00, 1.477278987220550e-05, 1.477146701080788e-05, 0.000000000000000e+00, 1.477198213440255e-05, 1.477146701080788e-05, 0.000000000000000e+00, 1.477198213440255e-05, 3.266165077737750e-03, 0.000000000000000e+00, 3.267491075456034e-03, 3.266283933231718e-03, 0.000000000000000e+00, 3.267704210664432e-03, 3.270100736130752e-03, 0.000000000000000e+00, 3.271036443996367e-03, 3.261874089704156e-03, 0.000000000000000e+00, 3.263057365411939e-03, 3.267332907216904e-03, 0.000000000000000e+00, 3.266244390065417e-03, 3.267332907216904e-03, 0.000000000000000e+00, 3.266244390065417e-03, 3.483132775919240e-01, 0.000000000000000e+00, 3.564533818401373e-01, 3.458501941252841e-01, 0.000000000000000e+00, 3.556980243972697e-01, 3.234191918679691e-01, 0.000000000000000e+00, 3.207956090336642e-01, 3.147804728669806e-01, 0.000000000000000e+00, 3.159965875804253e-01, 3.671925855587924e-01, 0.000000000000000e+00, 3.623731756918517e-01, 3.671925855587924e-01, 0.000000000000000e+00, 3.623731756918517e-01, 2.425999396986805e+01, 0.000000000000000e+00, 2.138871626721380e+01, 2.323706851606361e+01, 0.000000000000000e+00, 2.008651845236410e+01, 2.340477852199042e-01, 0.000000000000000e+00, 2.421236384387165e-01, 5.564794044973035e+01, 0.000000000000000e+00, 5.337382921345535e+01, 2.125001293838128e+01, 0.000000000000000e+00, 1.456855405919616e+02, 2.125001293838128e+01, 0.000000000000000e+00, 1.456855405919616e+02, 6.324502538132736e+05, 0.000000000000000e+00, 5.271019232889117e+05, 5.496562201585606e+05, 0.000000000000000e+00, 4.471116175230451e+05, 3.196986968457005e+03, 0.000000000000000e+00, 2.714719267750195e+03, 2.911560166641880e+06, 0.000000000000000e+00, 3.062094192421960e+06, 8.908245766567425e+05, 0.000000000000000e+00, 4.834051090088997e+06, 8.908245766567425e+05, 0.000000000000000e+00, 4.834051090088992e+06, 9.895369287559058e-04, 0.000000000000000e+00, 9.903118049001121e-04, 9.873989847942676e-04, 0.000000000000000e+00, 9.882466156845571e-04, 9.894232174613722e-04, 0.000000000000000e+00, 9.902443156129324e-04, 9.875894110496623e-04, 0.000000000000000e+00, 9.883686767713520e-04, 9.884439010640646e-04, 0.000000000000000e+00, 9.892788744178013e-04, 9.884439010640646e-04, 0.000000000000000e+00, 9.892788744178013e-04, 1.168469529968785e-02, 0.000000000000000e+00, 1.168629867297421e-02, 1.149870335363116e-02, 0.000000000000000e+00, 1.150629433148625e-02, 1.179271145974800e-02, 0.000000000000000e+00, 1.175268418338678e-02, 1.159309051298412e-02, 0.000000000000000e+00, 1.156136791108390e-02, 1.158131665453238e-02, 0.000000000000000e+00, 1.158740985550595e-02, 1.158131665453238e-02, 0.000000000000000e+00, 1.158740985550595e-02, 6.996440680864914e-01, 0.000000000000000e+00, 7.055358752167836e-01, 3.986966278867722e-01, 0.000000000000000e+00, 3.920557175600597e-01, 9.000052451373450e-01, 0.000000000000000e+00, 8.274071984888262e-01, 6.916826759390408e-01, 0.000000000000000e+00, 6.347238228854989e-01, 6.302976220372095e-01, 0.000000000000000e+00, 7.327066710862672e-01, 6.302976220372096e-01, 0.000000000000000e+00, 7.327066710862674e-01, 1.675776556058102e+02, 0.000000000000000e+00, 1.628428412687544e+02, 2.363286893929018e+01, 0.000000000000000e+00, 2.327138706993068e+01, 2.265472608219240e+02, 0.000000000000000e+00, 1.935385103685561e+02, 1.402585478405162e-02, 0.000000000000000e+00, 1.403258123869395e-02, 1.114833739851550e+02, 0.000000000000000e+00, 9.334293884176547e+01, 1.114833739851550e+02, 0.000000000000000e+00, 9.334293884176547e+01, 6.896246119618593e+06, 0.000000000000000e+00, 6.144157391435340e+06, 3.268601479903577e+06, 0.000000000000000e+00, 3.124571863316902e+06, 7.817219724160801e+06, 0.000000000000000e+00, 6.581441198317574e+06, 6.935893041367808e+02, 0.000000000000000e+00, 6.809561864073365e+02, 7.337213843636003e+06, 0.000000000000000e+00, 2.832798199981855e+06, 7.337213843636003e+06, 0.000000000000000e+00, 2.832798199981855e+06, 2.653013067933165e-01, 0.000000000000000e+00, 2.573771324712747e-01, 5.007182530913025e-01, 0.000000000000000e+00, 4.896750447557582e-01, 4.312088571430769e-01, 0.000000000000000e+00, 4.204174629436528e-01, 3.628691320156430e-01, 0.000000000000000e+00, 3.539705261206467e-01, 3.982710209888468e-01, 0.000000000000000e+00, 3.883875665156685e-01, 3.982710209888468e-01, 0.000000000000000e+00, 3.883875665156685e-01, 1.828129622790040e-01, 0.000000000000000e+00, 1.801293381316302e-01, 9.101289575998702e-01, 0.000000000000000e+00, 9.062076569708509e-01, 9.869856268263388e-01, 0.000000000000000e+00, 9.799994304864462e-01, 9.154688065280194e-01, 0.000000000000000e+00, 9.048560751662204e-01, 9.822087209297218e-01, 0.000000000000000e+00, 9.721809837800128e-01, 9.822087209297218e-01, 0.000000000000000e+00, 9.721809837800128e-01, 3.645849584052111e-01, 0.000000000000000e+00, 3.553666474927719e-01, 1.142269383606850e+01, 0.000000000000000e+00, 1.121240171805671e+01, 6.149725258664654e+00, 0.000000000000000e+00, 5.896038319468314e+00, 2.345095018610140e+00, 0.000000000000000e+00, 2.322702753584334e+00, 3.223486015872326e+00, 0.000000000000000e+00, 3.217602394098347e+00, 3.223486015872328e+00, 0.000000000000000e+00, 3.217602394098346e+00, 1.146016588251108e+00, 0.000000000000000e+00, 1.141302412659026e+00, 3.383343942871937e+03, 0.000000000000000e+00, 3.314830002033787e+03, 1.389957797964423e+03, 0.000000000000000e+00, 1.259186451761357e+03, 3.246781643675896e+00, 0.000000000000000e+00, 3.177025044255121e+00, 3.620451681580888e+02, 0.000000000000000e+00, 3.072885516476045e+02, 3.620451681580886e+02, 0.000000000000000e+00, 3.072885516476045e+02, 2.160802094642876e+05, 0.000000000000000e+00, 1.941850549004907e+05, 1.678524443151690e+08, 0.000000000000000e+00, 1.667067705549431e+08, 1.992427883915158e+07, 0.000000000000000e+00, 1.656041583904839e+07, 4.116905005353001e+02, 0.000000000000000e+00, 3.921081285660117e+02, 8.633543420441184e+06, 0.000000000000000e+00, 3.678861636060889e+06, 8.633543420441205e+06, 0.000000000000000e+00, 3.678861636060901e+06]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05