
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_mgga_x_gx_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-2.431243754719119e+01, -2.431251144868043e+01, -2.431292164272360e+01, -2.431182437371829e+01, -2.431238765548355e+01, -2.431238765548355e+01, -3.424164879326502e+00, -3.424385754955226e+00, -3.430579071820159e+00, -3.431807089495234e+00, -3.429860681355784e+00, -3.429860681355784e+00, -5.916494155496730e-01, -5.908760284089121e-01, -5.729819633734456e-01, -5.820710869989842e-01, -5.822075974163731e-01, -5.822075974163731e-01, -1.881465939839345e-01, -1.894225791137141e-01, -6.590063168732460e-01, -1.146710876741688e-01, -1.692638715875794e-01, -1.692638715875793e-01, -4.829031065269685e-03, -5.061382130114755e-03, -2.819681217221666e-02, -2.754083974965311e-03, -3.848960698481940e-03, -3.848960698481940e-03, -5.953920200630021e+00, -5.955038844633410e+00, -5.954017008348940e+00, -5.955002638639057e+00, -5.954463902653190e+00, -5.954463902653190e+00, -2.158712134249173e+00, -2.187589862730098e+00, -2.156233348254632e+00, -2.182479802468770e+00, -2.176231157213267e+00, -2.176231157213267e+00, -6.352105120290698e-01, -6.925227901789867e-01, -5.558459216281240e-01, -5.741762479319059e-01, -6.496138675163673e-01, -6.496138675163674e-01, -7.659621597185362e-02, -1.798177606385172e-01, -7.042537336560033e-02, -1.977529506276693e+00, -9.544793694119375e-02, -9.544793694119376e-02, -2.125859333933817e-03, -2.691968449559695e-03, -2.061923442104351e-03, -4.634435848890908e-02, -2.593010266606245e-03, -2.593010266606246e-03, -6.702604818132408e-01, -6.664935156241042e-01, -6.678184301180839e-01, -6.689092815971227e-01, -6.683610551103185e-01, -6.683610551103185e-01, -6.480707366577627e-01, -5.583053033595097e-01, -5.836801030288670e-01, -6.090374561217238e-01, -5.956791261211362e-01, -5.956791261211362e-01, -7.151196376685693e-01, -2.402820534641801e-01, -2.817345139668254e-01, -3.581860003370334e-01, -3.209989224651562e-01, -3.209989224651562e-01, -4.924730368850544e-01, -2.728163347673222e-02, -3.694044069805717e-02, -3.535691656406191e-01, -6.084423285153225e-02, -6.084423285153226e-02, -6.735098573539577e-03, -7.194395026259479e-04, -1.513278483771800e-03, -5.852342258736307e-02, -2.380624087696805e-03, -2.380624087696803e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_mgga_x_gx_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-3.084795263256600e+01, -3.084778048712448e+01, -3.084811108766134e+01, -3.084789558214352e+01, -3.084853946039262e+01, -3.084853142559116e+01, -3.084675062269192e+01, -3.084625110723074e+01, -3.084805057928683e+01, -3.084710494738891e+01, -3.084805057928683e+01, -3.084710494738891e+01, -5.177362581350004e+00, -5.179451464168364e+00, -5.177041813411640e+00, -5.179419020747412e+00, -5.173065623463017e+00, -5.172943091499254e+00, -5.166590910554113e+00, -5.168451000573770e+00, -5.177055137543686e+00, -5.165166861508529e+00, -5.177055137543686e+00, -5.165166861508529e+00, -8.324621384795365e-01, -8.376926809537057e-01, -8.306194730796828e-01, -8.368538627379348e-01, -8.054308430930189e-01, -7.983667971223686e-01, -8.114164458720664e-01, -8.146048952357910e-01, -8.451397902360990e-01, -7.698674632294844e-01, -8.451397902360990e-01, -7.698674632294844e-01, -1.084513652766880e-01, -1.337573358244697e-01, -1.170789191492349e-01, -1.460443319528950e-01, -9.023698733256250e-01, -9.473280170732847e-01, -1.402896641074499e-01, -1.260479170690536e-01, -1.142159864345017e-01, -1.030036302624548e-01, -1.142159864345017e-01, -1.030036302624547e-01, -4.039940813295510e-03, 6.960517802737054e-03, -5.568475430121306e-03, -3.423383417374464e-03, -3.415003818538961e-02, -3.478904930543959e-02, -3.698970860167774e-03, -3.636145887408222e-03, -5.343215875626282e-03, -3.125670011956926e-03, -5.343215875626296e-03, -3.125670011956927e-03, -7.629677214552345e+00, -7.627709099174002e+00, -7.633645328934787e+00, -7.631543143268826e+00, -7.629879388407903e+00, -7.627827682157368e+00, -7.633284983855706e+00, -7.631311401979158e+00, -7.631724515660331e+00, -7.629637853115510e+00, -7.631724515660331e+00, -7.629637853115510e+00, -2.452923366489404e+00, -2.452295600537548e+00, -2.481723937155027e+00, -2.480352508062913e+00, -2.400592323390239e+00, -2.415458606948989e+00, -2.425964195534322e+00, -2.440687091939191e+00, -2.506626167571154e+00, -2.472365124133996e+00, -2.506626167571154e+00, -2.472365124133996e+00, -8.494651989539295e-01, -8.467982810537580e-01, -9.679941648085363e-01, -9.682523469408885e-01, -7.808824753929431e-01, -8.020412328507976e-01, -8.702232919307697e-01, -8.818940299944202e-01, -8.884222988036395e-01, -8.452200906633126e-01, -8.884222988036395e-01, -8.452200906633127e-01, -8.770491104801163e-02, -9.066444079380644e-02, -1.926267063858497e-02, -1.914844196999547e-02, -8.326901230642661e-02, -8.415342673800147e-02, -2.978699709088222e+00, -2.977337522097382e+00, -8.901740164882647e-02, -8.168213791724721e-02, -8.901740164882636e-02, -8.168213791724691e-02, -2.776842654030625e-03, -2.885808886057298e-03, -3.561267606921179e-03, -3.615262709451901e-03, -2.660945946951170e-03, -2.817989322534501e-03, -5.858700727901374e-02, -5.400847259017255e-02, -2.720095596403397e-03, -3.567170125599913e-03, -2.720095596403398e-03, -3.567170125599912e-03, -8.934994154965330e-01, -8.970741073832944e-01, -8.787418292548764e-01, -8.824319997164506e-01, -8.839252495786698e-01, -8.876064284381099e-01, -8.882465819686346e-01, -8.918321053228029e-01, -8.860877811677422e-01, -8.897190705180317e-01, -8.860877811677422e-01, -8.897190705180317e-01, -8.737268334010658e-01, -8.765370844285897e-01, -6.583528149511862e-01, -6.624340788486158e-01, -7.232720481594762e-01, -7.276129710140502e-01, -7.860000809173462e-01, -7.890722521139930e-01, -7.550612797829422e-01, -7.581853600684136e-01, -7.550612797829422e-01, -7.581853600684137e-01, -1.017389159199021e+00, -1.018177216387661e+00, -1.392728215133203e-01, -1.427513621319009e-01, -2.491694506410423e-01, -2.572671053929372e-01, -4.609165011229914e-01, -4.645852127893002e-01, -3.556224511978985e-01, -3.564747937939690e-01, -3.556224511978985e-01, -3.564747937939691e-01, -6.078867850061149e-01, -6.136878441817790e-01, -2.249226683429789e-02, -2.694372076788950e-02, -4.492154935139112e-02, -4.593257069241700e-02, -4.647926227011058e-01, -4.698113661486699e-01, -6.604912806193736e-02, -6.759445594805717e-02, -6.604912806193741e-02, -6.759445594805720e-02, -8.806839021236596e-03, -9.126871905633084e-03, -9.581559510130764e-04, -9.603456530509827e-04, -1.946171227370981e-03, -2.069667527515848e-03, -5.902887076620364e-02, -1.655817213498272e-02, -2.576545109968189e-03, -3.298367201877172e-03, -2.576545109968186e-03, -3.298367201877178e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gx_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-2.391632047437558e-08, 0.000000000000000e+00, -2.391620635629717e-08, -2.391600773365999e-08, 0.000000000000000e+00, -2.391597919449412e-08, -2.391491729698753e-08, 0.000000000000000e+00, -2.391444194230640e-08, -2.391844923037190e-08, 0.000000000000000e+00, -2.391894355459556e-08, -2.391613758180106e-08, 0.000000000000000e+00, -2.391707800363186e-08, -2.391613758180106e-08, 0.000000000000000e+00, -2.391707800363186e-08, -4.657031274103915e-05, 0.000000000000000e+00, -4.665696161468569e-05, -4.654500936366923e-05, 0.000000000000000e+00, -4.664672126176607e-05, -4.613630706723012e-05, 0.000000000000000e+00, -4.608000336780951e-05, -4.598924405959911e-05, 0.000000000000000e+00, -4.604582771718203e-05, -4.665017977509291e-05, 0.000000000000000e+00, -4.568893629849172e-05, -4.665017977509291e-05, 0.000000000000000e+00, -4.568893629849172e-05, -4.165581607607781e-03, 0.000000000000000e+00, -4.184757528913859e-03, -4.175210888640184e-03, 0.000000000000000e+00, -4.178217782452981e-03, -4.207007174331051e-03, 0.000000000000000e+00, -4.423588779833686e-03, -4.790106529007970e-03, 0.000000000000000e+00, -4.858164294329843e-03, -3.982759858753511e-03, 0.000000000000000e+00, -6.078747932482285e-03, -3.982759858753511e-03, 0.000000000000000e+00, -6.078747932482285e-03, -6.490314970792352e+00, 0.000000000000000e+00, -5.203643440448525e+00, -6.334794596885026e+00, 0.000000000000000e+00, -4.874548337173908e+00, -6.427201742434301e-04, 0.000000000000000e+00, -7.733824214572523e-04, -1.423517369130984e+00, 0.000000000000000e+00, -3.153781480135978e+00, -5.656508058533397e+00, 0.000000000000000e+00, -7.004885942921725e-01, -5.656508058533393e+00, 0.000000000000000e+00, -7.004885942921706e-01, -1.911155581936351e+03, 0.000000000000000e+00, -9.891722928197929e+03, -7.304049415548405e+02, 0.000000000000000e+00, -2.274247839680234e+03, -9.970035075789175e+00, 0.000000000000000e+00, -1.387753416915119e+01, -1.551215465764053e+01, 0.000000000000000e+00, -2.139254879984988e+01, -2.066049065501675e+02, 0.000000000000000e+00, -1.080650867789274e+01, -2.066049065501471e+02, 0.000000000000000e+00, -1.080650867789818e+01, -6.754876448052904e-06, 0.000000000000000e+00, -6.761285400670301e-06, -6.749818408096983e-06, 0.000000000000000e+00, -6.756393951596132e-06, -6.754541566655461e-06, 0.000000000000000e+00, -6.761078606452921e-06, -6.750202809122214e-06, 0.000000000000000e+00, -6.756636197889770e-06, -6.752326182904743e-06, 0.000000000000000e+00, -6.758834944794050e-06, -6.752326182904743e-06, 0.000000000000000e+00, -6.758834944794050e-06, -3.466765021383704e-04, 0.000000000000000e+00, -3.463751429543360e-04, -3.349714704344778e-04, 0.000000000000000e+00, -3.350872483099813e-04, -3.539519053425182e-04, 0.000000000000000e+00, -3.516055011902684e-04, -3.435740849561684e-04, 0.000000000000000e+00, -3.411656112628901e-04, -3.347176541137634e-04, 0.000000000000000e+00, -3.396101367963040e-04, -3.347176541137634e-04, 0.000000000000000e+00, -3.396101367963040e-04, -4.351648590724980e-02, 0.000000000000000e+00, -4.384374668896614e-02, -3.130983079086280e-02, 0.000000000000000e+00, -3.116986543306983e-02, -7.727583115165848e-02, 0.000000000000000e+00, -5.904561775092593e-02, -6.568987711939726e-02, 0.000000000000000e+00, -5.230250070993668e-02, -3.666813394513909e-02, 0.000000000000000e+00, -4.576821331606317e-02, -3.666813394513908e-02, 0.000000000000000e+00, -4.576821331606316e-02, -3.664445279609797e+00, 0.000000000000000e+00, -3.166334661137761e+00, -6.167458602502897e+00, 0.000000000000000e+00, -6.127658012933575e+00, -2.945883753344211e+00, 0.000000000000000e+00, -3.915236154942320e+00, -4.159944690420616e-04, 0.000000000000000e+00, -4.166631750593953e-04, -5.778621310836186e+00, 0.000000000000000e+00, -8.693764287495283e+00, -5.778621310836201e+00, 0.000000000000000e+00, -8.693764287495327e+00, -1.158227813248423e+00, 0.000000000000000e+00, -1.192190618072278e+00, -4.234734000587330e+00, 0.000000000000000e+00, -3.186097108885125e+00, -9.986599972842446e+01, 0.000000000000000e+00, -9.967221737584019e+01, -3.694005711628439e+00, 0.000000000000000e+00, -9.162278828022913e+00, -1.293847963689725e+00, 0.000000000000000e+00, -1.250035144912587e+03, -1.293847963689158e+00, 0.000000000000000e+00, -1.250035144912598e+03, -4.154993632691328e-02, 0.000000000000000e+00, -4.090981443570350e-02, -4.251769630035639e-02, 0.000000000000000e+00, -4.185636382584036e-02, -4.217507139859366e-02, 0.000000000000000e+00, -4.151917187713235e-02, -4.189246872116096e-02, 0.000000000000000e+00, -4.124691155987392e-02, -4.203368367989400e-02, 0.000000000000000e+00, -4.138300019407787e-02, -4.203368367989400e-02, 0.000000000000000e+00, -4.138300019407787e-02, -4.649014764050675e-02, 0.000000000000000e+00, -4.586842958354952e-02, -8.560931536070736e-02, 0.000000000000000e+00, -8.413445658458177e-02, -7.127990842067890e-02, 0.000000000000000e+00, -7.002020370363980e-02, -5.977604366940888e-02, 0.000000000000000e+00, -5.893954635644882e-02, -6.533227079231260e-02, 0.000000000000000e+00, -6.435343376499526e-02, -6.533227079231260e-02, 0.000000000000000e+00, -6.435343376499526e-02, -2.652021029967334e-02, 0.000000000000000e+00, -2.615857593706951e-02, -2.240331998630336e+00, 0.000000000000000e+00, -2.190988805579660e+00, -1.117724796153304e+00, 0.000000000000000e+00, -1.071232362114340e+00, -3.903361169127115e-01, 0.000000000000000e+00, -3.835470992633526e-01, -6.354162045589742e-01, 0.000000000000000e+00, -6.362769512274092e-01, -6.354162045589747e-01, 0.000000000000000e+00, -6.362769512274099e-01, -1.242312528256511e-01, 0.000000000000000e+00, -1.200936714142939e-01, -5.302056729803098e+01, 0.000000000000000e+00, -3.630540978125618e+01, -6.695998361477668e+00, 0.000000000000000e+00, -7.374083117245602e+00, -4.577126794898037e-01, 0.000000000000000e+00, -4.018864073979697e-01, -7.847232938475440e+00, 0.000000000000000e+00, -9.709402616558599e+00, -7.847232938475408e+00, 0.000000000000000e+00, -9.709402616558600e+00, -2.640299035096688e+00, 0.000000000000000e+00, -2.320198025852115e+00, -7.449855448913774e-01, 0.000000000000000e+00, -5.152725348161804e-01, -1.815889250188113e+02, 0.000000000000000e+00, -1.783169006210013e+02, -1.195028132101683e+01, 0.000000000000000e+00, -3.822841923790847e+01, -4.518907733920247e+00, 0.000000000000000e+00, -1.235559030721203e+03, -4.518907733920230e+00, 0.000000000000000e+00, -1.235559030721153e+03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gx_BrOH_cation_2_vlapl():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vlapl"].flatten()
    ref_tgt = [0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_mgga_x_gx_BrOH_cation_2_vtau():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("mgga_x_gx", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vtau"].flatten()
    ref_tgt = [1.849520579494709e-03, 1.849504609507420e-03, 1.849514320013752e-03, 1.849500064830939e-03, 1.849490876110744e-03, 1.849467433747122e-03, 1.849561473938681e-03, 1.849557432752622e-03, 1.849516948524766e-03, 1.849518955368838e-03, 1.849516948524766e-03, 1.849518955368838e-03, 1.554226432033317e-02, 1.557054495326680e-02, 1.553396532733620e-02, 1.556718309380366e-02, 1.540016584323067e-02, 1.538232183332197e-02, 1.535401618884289e-02, 1.537355452443066e-02, 1.556111013970134e-02, 1.525758184167165e-02, 1.556111013970134e-02, 1.525758184167165e-02, 9.916363231415331e-03, 1.008094588640488e-02, 9.892952760132358e-03, 1.004544138238464e-02, 9.358483436355730e-03, 9.638300291286752e-03, 1.077594871849997e-02, 1.100023585907839e-02, 9.816784890415039e-03, 1.231201092882927e-02, 9.816784890415039e-03, 1.231201092882927e-02, 2.522792550680095e-01, 2.267999066911431e-01, 2.554492031103202e-01, 2.239051200787946e-01, 2.319327646078950e-03, 3.081862917785257e-03, 2.515151380676253e-02, 5.806179325141415e-02, 2.503156633261481e-01, 4.786364592416039e-03, 2.503156633261479e-01, 4.786364592416030e-03, 3.021827292413394e-03, 1.876624150085058e-02, 1.328839454357605e-03, 5.086532009834263e-03, 3.118403621358698e-03, 5.111610619313017e-03, 5.327780856441160e-06, 6.986247737868806e-06, 2.319254566333666e-04, 2.235497378874266e-06, 2.319254566333401e-04, 2.235497378875480e-06, 7.568685131283173e-03, 7.570423386267680e-03, 7.567255605964630e-03, 7.569042027196456e-03, 7.568568108902123e-03, 7.570348856610846e-03, 7.567342762556422e-03, 7.569095168739396e-03, 7.567981403382883e-03, 7.569734282019194e-03, 7.567981403382883e-03, 7.569734282019194e-03, 2.062304912892364e-02, 2.060199845449351e-02, 2.037027991826832e-02, 2.036068861345547e-02, 2.066783736089365e-02, 2.063754111258791e-02, 2.044999717678024e-02, 2.041660599952158e-02, 2.043881045245666e-02, 2.046629890179648e-02, 2.043881045245666e-02, 2.046629890179648e-02, 7.002799625741857e-02, 6.999573630987993e-02, 6.429707447378159e-02, 6.409686205539858e-02, 9.281805709078739e-02, 7.903434281686066e-02, 8.910773748818240e-02, 7.825753374041293e-02, 6.638155986147794e-02, 7.107527892722951e-02, 6.638155986147792e-02, 7.107527892722948e-02, 2.180798790117712e-02, 1.938785530026609e-02, 2.535215776462100e-01, 2.556755281184694e-01, 1.297839098589238e-02, 2.018142616632623e-02, 2.336792959307011e-02, 2.337396145438384e-02, 5.151964491295001e-02, 9.214198680934779e-02, 5.151964491295016e-02, 9.214198680934824e-02, 1.679504745814956e-07, 1.940364703099858e-07, 1.295579772324650e-06, 1.019690766982176e-06, 1.277512998169843e-05, 1.514443600695250e-05, 5.323016849452069e-03, 1.344777054147941e-02, 1.763404992209296e-07, 4.412722041740211e-04, 1.763404992208205e-07, 4.412722041740267e-04, 6.718229555301380e-02, 6.691735882599327e-02, 6.756683866098628e-02, 6.729808932122225e-02, 6.743152101408235e-02, 6.716320149828564e-02, 6.731917455558373e-02, 6.705381778868989e-02, 6.737549309624240e-02, 6.710863118255134e-02, 6.737549309624240e-02, 6.710863118255136e-02, 6.927979907568430e-02, 6.903342267316863e-02, 8.057647227306092e-02, 8.022198992962930e-02, 7.702367009189978e-02, 7.667429925991251e-02, 7.374314678466863e-02, 7.348092066305938e-02, 7.540936427796399e-02, 7.511040695885279e-02, 7.540936427796399e-02, 7.511040695885278e-02, 6.264165006119303e-02, 6.209000136824843e-02, 1.855072136824416e-01, 1.845910556213173e-01, 1.592881621102975e-01, 1.573020571788321e-01, 1.290629363890150e-01, 1.289334544342108e-01, 1.394045452978188e-01, 1.394457059462081e-01, 1.394045452978189e-01, 1.394457059462082e-01, 9.089084678871230e-02, 8.969132971738886e-02, 1.567033926574905e-02, 1.095190505907747e-02, 4.816666342174066e-03, 5.855163309082943e-03, 1.336455990354866e-01, 1.227340306626712e-01, 2.164331577608093e-02, 3.152555260274179e-02, 2.164331577608084e-02, 3.152555260274180e-02, 1.221906615495747e-05, 1.194838265556870e-05, 4.438335990865656e-09, 3.090891467268816e-09, 9.113952186175494e-06, 1.076765836717375e-05, 2.898818005582008e-02, 9.735617983906776e-02, 5.234128615025935e-07, 3.358536211878393e-04, 5.234128615020072e-07, 3.358536211878251e-04]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05