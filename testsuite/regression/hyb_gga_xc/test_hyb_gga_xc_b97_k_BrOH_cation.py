
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b97_k_BrOH_cation_2_zk():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.189930967296259e+01, -1.189930008781459e+01, -1.189933584746348e+01, -1.189947683882792e+01, -1.189939182079490e+01, -1.189939182079490e+01, -2.067596888377043e+00, -2.067548937053164e+00, -2.066489920050946e+00, -2.068715365329419e+00, -2.067588261275695e+00, -2.067588261275695e+00, -4.433579686966099e-01, -4.434658010378266e-01, -4.472593303183111e-01, -4.501725568761268e-01, -4.462957698454761e-01, -4.462957698454761e-01, -1.330656673624308e-01, -1.344377819183907e-01, -5.082114045303379e-01, -1.035752463093720e-01, -1.296112381609313e-01, -1.296112381609313e-01, -4.945181597782088e-03, -5.190424707154425e-03, -2.620858995086332e-02, -2.909751285967983e-03, -2.760402462476796e-03, -2.760402462476798e-03, -2.867854286232385e+00, -2.866432883218176e+00, -2.867799932470224e+00, -2.866544936072813e+00, -2.867127669734341e+00, -2.867127669734341e+00, -1.337136006715889e+00, -1.341823258605154e+00, -1.341209747997770e+00, -1.345447042100653e+00, -1.337623516619750e+00, -1.337623516619750e+00, -3.700780103811699e-01, -3.860921985580739e-01, -3.466841529814929e-01, -3.480270490554258e-01, -3.738015515838608e-01, -3.738015515838609e-01, -6.978615967370121e-02, -1.364502905200985e-01, -6.431413889953470e-02, -1.053620001174975e+00, -8.413248309793529e-02, -8.413248309793529e-02, -2.253098696159682e-03, -2.846105308163294e-03, -2.179558874473975e-03, -4.249347755407595e-02, -2.362920917783715e-03, -2.362920917783715e-03, -3.525540548949579e-01, -3.582330373765430e-01, -3.574642017694229e-01, -3.561057747720510e-01, -3.568986864017656e-01, -3.568986864017656e-01, -3.374686379296987e-01, -3.278340334229826e-01, -3.317502894824571e-01, -3.370711505579900e-01, -3.339307350445061e-01, -3.339307350445061e-01, -4.032578018051508e-01, -1.698380532918646e-01, -1.971227243074057e-01, -2.374292093758155e-01, -2.165106522818376e-01, -2.165106522818376e-01, -3.034481483112281e-01, -2.514127264195674e-02, -3.391079035975705e-02, -2.240615448293218e-01, -5.532908082573860e-02, -5.532908082573863e-02, -6.903347607376751e-03, -7.724592451774379e-04, -1.604952915113038e-03, -5.155754653567629e-02, -2.230773219186612e-03, -2.230773219186613e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b97_k_BrOH_cation_2_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-1.201557926953324e+01, -1.201556735754351e+01, -1.201564943355683e+01, -1.201561941130196e+01, -1.201584261810129e+01, -1.201589911497329e+01, -1.201503677956775e+01, -1.201488815086375e+01, -1.201561128914318e+01, -1.201527558688887e+01, -1.201561128914318e+01, -1.201527558688887e+01, -2.058702338418250e+00, -2.058776684791328e+00, -2.058734268675891e+00, -2.058811877777829e+00, -2.059537450761770e+00, -2.059659850735600e+00, -2.058468984687187e+00, -2.058591579154797e+00, -2.058452997961424e+00, -2.059198954245499e+00, -2.058452997961424e+00, -2.059198954245499e+00, -4.614890405812160e-01, -4.592624309152856e-01, -4.611160175906668e-01, -4.582978309521801e-01, -4.441740411734397e-01, -4.511051316909018e-01, -4.517345709596966e-01, -4.494661264935816e-01, -4.293446239808484e-01, -4.992034515863895e-01, -4.293446239808484e-01, -4.992034515863895e-01, -1.536345679902875e-01, -1.517862988913797e-01, -1.540489607308586e-01, -1.513657591530590e-01, -5.308576193954091e-01, -5.274852893096798e-01, -1.377109969746124e-01, -1.403380006186249e-01, -1.577205598857093e-01, -4.626938403489194e-02, -1.577205598857095e-01, -4.626938403489126e-02, -6.985721628791856e-03, -6.171460826702265e-03, -7.393077344970792e-03, -6.435317847379640e-03, -3.588573997423098e-02, -3.562165944922750e-02, -3.788507606414367e-03, -3.933366602099940e-03, -2.868760151340318e-03, -8.249121519635454e-03, -2.868760151340327e-03, -8.249121519635401e-03, -3.155507106999640e+00, -3.154849163062728e+00, -3.157882136930311e+00, -3.157157178467937e+00, -3.155629789969515e+00, -3.154929038701292e+00, -3.157677768137425e+00, -3.157015385557759e+00, -3.156725002568661e+00, -3.156011143317194e+00, -3.156725002568661e+00, -3.156011143317194e+00, -1.216463584320643e+00, -1.216441073355146e+00, -1.221006014260781e+00, -1.220941751048222e+00, -1.221954125955711e+00, -1.217591274997504e+00, -1.225317473856661e+00, -1.221281102801201e+00, -1.214352281016696e+00, -1.221672731348075e+00, -1.214352281016696e+00, -1.221672731348075e+00, -4.342919831390709e-01, -4.341100129759858e-01, -4.852025830368989e-01, -4.845504834679736e-01, -4.016253315419671e-01, -4.031885471729366e-01, -4.383427701794830e-01, -4.420459022517502e-01, -4.464322302166712e-01, -4.422284274947779e-01, -4.464322302166713e-01, -4.422284274947780e-01, -1.029762875508615e-01, -1.054977961090144e-01, -1.735722008712073e-01, -1.742884748243018e-01, -9.094387612928743e-02, -1.007154435617992e-01, -1.290187402213377e+00, -1.289806184633475e+00, -1.107349173866781e-01, -1.261869410293660e-01, -1.107349173866781e-01, -1.261869410293660e-01, -3.133040964377938e-03, -2.865970413140258e-03, -3.840483154678890e-03, -3.712648168523141e-03, -3.102725638191175e-03, -2.720307553693034e-03, -6.154712625096797e-02, -6.148311591007575e-02, -4.894643520887783e-03, -2.476578621541992e-03, -4.894643520887826e-03, -2.476578621541992e-03, -4.309945558308510e-01, -4.309577472380732e-01, -4.505986636082243e-01, -4.505576211673075e-01, -4.461648761566483e-01, -4.459769462100447e-01, -4.402405923349883e-01, -4.401374612558652e-01, -4.434542774395228e-01, -4.432966737398804e-01, -4.434542774395228e-01, -4.432966737398804e-01, -4.142264614392590e-01, -4.146328116213214e-01, -3.551336428650952e-01, -3.548085667048189e-01, -3.826871424149252e-01, -3.827407649350260e-01, -4.164881979308820e-01, -4.167362954487640e-01, -3.991632567655471e-01, -3.993599659096662e-01, -3.991632567655471e-01, -3.993599659096662e-01, -5.065083088141933e-01, -5.058653582091034e-01, -1.980382361298270e-01, -1.979914883401625e-01, -2.146294969272919e-01, -2.123458453971832e-01, -2.629574818347657e-01, -2.624899216110881e-01, -2.310803184559413e-01, -2.303696520451460e-01, -2.310803184559412e-01, -2.303696520451460e-01, -3.310339916144266e-01, -3.302606636401260e-01, -3.408089421069488e-02, -3.401479889544544e-02, -4.692179234685620e-02, -4.799530690254294e-02, -2.633804807397839e-01, -2.623065702207861e-01, -7.507074170386220e-02, -8.766189242678753e-02, -7.507074170386235e-02, -8.766189242678768e-02, -9.437159441761791e-03, -8.824153717210834e-03, -1.031752959983151e-03, -1.025916697712168e-03, -2.306887495214073e-03, -1.990356869608437e-03, -7.502376607213998e-02, -7.699766175952127e-02, -4.404979697906061e-03, -2.360185818186197e-03, -4.404979697906042e-03, -2.360185818186198e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b97_k_BrOH_cation_2_vsigma():
    # Prepare the input
    inp = test_data["BrOH_cation"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_k", 2)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-1.934346058316781e-08, 0.000000000000000e+00, -1.934357087911492e-08, -1.934337808168745e-08, 0.000000000000000e+00, -1.934350816309084e-08, -1.934283437959282e-08, 0.000000000000000e+00, -1.934282167228464e-08, -1.934380803447989e-08, 0.000000000000000e+00, -1.934402043429301e-08, -1.934345369124197e-08, 0.000000000000000e+00, -1.934331961764270e-08, -1.934345369124197e-08, 0.000000000000000e+00, -1.934331961764270e-08, -2.186597229999162e-05, 0.000000000000000e+00, -2.187644607000906e-05, -2.186662839095033e-05, 0.000000000000000e+00, -2.187784962926390e-05, -2.189018858826919e-05, 0.000000000000000e+00, -2.189769044030832e-05, -2.183550206437084e-05, 0.000000000000000e+00, -2.184481063692855e-05, -2.187452432036913e-05, 0.000000000000000e+00, -2.186628125483862e-05, -2.187452432036913e-05, 0.000000000000000e+00, -2.186628125483862e-05, -8.891304629356108e-03, 0.000000000000000e+00, -9.137970278277669e-03, -8.864442830940700e-03, 0.000000000000000e+00, -9.171129322907442e-03, -9.051593975979721e-03, 0.000000000000000e+00, -8.506613021829895e-03, -8.496494119210664e-03, 0.000000000000000e+00, -8.672370874797324e-03, -1.081650074861939e-02, 0.000000000000000e+00, -5.526852228123740e-03, -1.081650074861939e-02, 0.000000000000000e+00, -5.526852228123740e-03, -3.354887342105161e-01, 0.000000000000000e+00, -4.099744002248557e-01, -3.526879041066077e-01, 0.000000000000000e+00, -4.396322936891837e-01, -5.184894729597607e-03, 0.000000000000000e+00, -5.428033935254389e-03, 1.285155758817818e-02, 0.000000000000000e+00, 3.667228196273174e-02, -3.215683504297435e-01, 0.000000000000000e+00, -4.480697989239101e+00, -3.215683504297414e-01, 0.000000000000000e+00, -4.480697989239137e+00, -1.345496930770867e+00, 0.000000000000000e+00, 3.220492207383848e+00, -1.781860650994912e+00, 0.000000000000000e+00, 3.729046606528199e+00, 8.847509456761561e-01, 0.000000000000000e+00, 2.591414847402330e+00, 1.387026101307845e+00, 0.000000000000000e+00, -1.130831151732843e-01, 9.497574854774523e+00, 0.000000000000000e+00, -9.179803205080395e+01, 9.497574854775237e+00, 0.000000000000000e+00, -9.179803205098455e+01, -5.205945250265464e-06, 0.000000000000000e+00, -5.210127845383862e-06, -5.207942642199363e-06, 0.000000000000000e+00, -5.212015404532225e-06, -5.206024541928266e-06, 0.000000000000000e+00, -5.210146283339753e-06, -5.207705759269919e-06, 0.000000000000000e+00, -5.211896056135313e-06, -5.207012115391400e-06, 0.000000000000000e+00, -5.211074821921686e-06, -5.207012115391400e-06, 0.000000000000000e+00, -5.211074821921686e-06, -1.304868953966223e-04, 0.000000000000000e+00, -1.305018338207202e-04, -1.291434981553886e-04, 0.000000000000000e+00, -1.291980592880905e-04, -1.269511812623632e-04, 0.000000000000000e+00, -1.284941772852738e-04, -1.259570914330151e-04, 0.000000000000000e+00, -1.273767537163342e-04, -1.324114162170141e-04, 0.000000000000000e+00, -1.295909336198578e-04, -1.324114162170141e-04, 0.000000000000000e+00, -1.295909336198578e-04, -1.389304558104741e-02, 0.000000000000000e+00, -1.390906846342503e-02, -1.013319393976924e-02, 0.000000000000000e+00, -1.058742266504326e-02, -1.853832326244896e-02, 0.000000000000000e+00, -1.843701170583572e-02, -9.168427784408717e-03, 0.000000000000000e+00, -1.041281875074994e-02, -1.252576693061132e-02, 0.000000000000000e+00, -1.263164101740874e-02, -1.252576693061138e-02, 0.000000000000000e+00, -1.263164101740875e-02, 7.694212667187268e-01, 0.000000000000000e+00, 8.768795675067559e-01, -7.634476670512366e-02, 0.000000000000000e+00, -7.657017410352926e-02, 6.987887193792882e-01, 0.000000000000000e+00, 1.221181696307176e+00, -2.613136324083227e-04, 0.000000000000000e+00, -2.618489084388063e-04, 1.871474800377937e-01, 0.000000000000000e+00, 5.492363582279244e-01, 1.871474800377937e-01, 0.000000000000000e+00, 5.492363582279244e-01, -4.250269292109072e-01, 0.000000000000000e+00, 1.511094829986186e+00, 5.507556204273142e-01, 0.000000000000000e+00, 9.268976230234317e-01, -1.731859835282108e+01, 0.000000000000000e+00, 2.193540701231209e+01, 1.898241941235153e+00, 0.000000000000000e+00, 1.880171716057688e+00, -6.412195013653830e+01, 0.000000000000000e+00, 2.565034790328319e+01, -6.412195013625258e+01, 0.000000000000000e+00, 2.565034790333005e+01, -7.028586155740567e-02, 0.000000000000000e+00, -7.184856366463385e-02, -1.372778937967225e-02, 0.000000000000000e+00, -1.428959876196743e-02, -2.162925089243105e-02, 0.000000000000000e+00, -2.252853402552146e-02, -3.523697687890238e-02, 0.000000000000000e+00, -3.621018866365289e-02, -2.732427273433161e-02, 0.000000000000000e+00, -2.827753421774670e-02, -2.732427273433161e-02, 0.000000000000000e+00, -2.827753421774670e-02, -1.547377511493526e-01, 0.000000000000000e+00, -1.535646254674620e-01, -2.692433621088289e-02, 0.000000000000000e+00, -2.700037502777808e-02, -2.231612919519815e-02, 0.000000000000000e+00, -2.234677583811512e-02, -1.402759944722465e-02, 0.000000000000000e+00, -1.409412623487089e-02, -1.868700163273474e-02, 0.000000000000000e+00, -1.870547079049671e-02, -1.868700163273474e-02, 0.000000000000000e+00, -1.870547079049671e-02, -7.963557855635350e-03, 0.000000000000000e+00, -8.368841563189401e-03, -1.217422161724452e-01, 0.000000000000000e+00, -1.247483674182513e-01, -1.276402958423397e-01, 0.000000000000000e+00, -1.352447424908940e-01, -8.579176954259279e-02, 0.000000000000000e+00, -8.641574745373438e-02, -1.181324541757339e-01, 0.000000000000000e+00, -1.198348415015976e-01, -1.181324541757337e-01, 0.000000000000000e+00, -1.198348415015974e-01, -3.550225270160324e-02, 0.000000000000000e+00, -3.584255322043405e-02, 1.556486007453668e+00, 0.000000000000000e+00, 1.757279473830636e+00, 1.225916066416599e+00, 0.000000000000000e+00, 2.126523582586016e+00, -8.855001595968046e-02, 0.000000000000000e+00, -9.196360542116608e-02, 6.881949032087104e-01, 0.000000000000000e+00, 2.306712267897126e+00, 6.881949032087088e-01, 0.000000000000000e+00, 2.306712267897134e+00, 2.621389622262706e-02, 0.000000000000000e+00, 2.309741850507439e+00, -1.349858712189674e+01, 0.000000000000000e+00, 2.079385573118980e+01, -7.145728651835799e+00, 0.000000000000000e+00, 7.688327128567336e+00, 1.578825686143594e+00, 0.000000000000000e+00, 1.860763411576516e+00, -6.868362611617117e+01, 0.000000000000000e+00, 2.041998431847202e+01, -6.868362611615547e+01, 0.000000000000000e+00, 2.041998431843868e+01]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05