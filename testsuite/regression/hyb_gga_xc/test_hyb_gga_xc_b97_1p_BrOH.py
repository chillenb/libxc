
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_hyb_gga_xc_b97_1p_BrOH_1_zk():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_1p", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-1.789831597724450e+01, -1.789834705853236e+01, -1.789853508149459e+01, -1.789802514165842e+01, -1.789833195582794e+01, -1.789833195582794e+01, -3.004092127123373e+00, -3.004079855803860e+00, -3.003864797308972e+00, -3.004860942686951e+00, -3.004098365230582e+00, -3.004098365230582e+00, -6.252361969314659e-01, -6.251029416999532e-01, -6.241862011813183e-01, -6.279736373600875e-01, -6.251848622929239e-01, -6.251848622929239e-01, -2.000871278131496e-01, -2.011377068539834e-01, -7.296756510652243e-01, -1.666653426264055e-01, -2.003799408399860e-01, -2.003799408399860e-01, -1.266286066109649e-02, -1.334146881323992e-02, -6.384890777951642e-02, -5.584382449174830e-03, -1.318398889810771e-02, -1.318398889810771e-02, -4.392190813836693e+00, -4.392404996442135e+00, -4.392216391024025e+00, -4.392382852392415e+00, -4.392295040336617e+00, -4.392295040336617e+00, -1.817556321491846e+00, -1.825662006650817e+00, -1.818815811659551e+00, -1.825050999447362e+00, -1.821667789132035e+00, -1.821667789132035e+00, -5.309550968126383e-01, -5.659285955627030e-01, -5.060982763722097e-01, -5.197739016774964e-01, -5.486207620763571e-01, -5.486207620763571e-01, -1.365314790861321e-01, -2.188778159173995e-01, -1.334965932859891e-01, -1.641289701945384e+00, -1.486947861803560e-01, -1.486947861803560e-01, -5.367520794087520e-03, -6.219255400027569e-03, -4.527619451380816e-03, -8.864578933272076e-02, -5.611508424358419e-03, -5.611508424358420e-03, -5.336544898788166e-01, -5.360106150489053e-01, -5.359218933582742e-01, -5.354218973810869e-01, -5.357346725666869e-01, -5.357346725666869e-01, -5.154915303607966e-01, -4.724812710122026e-01, -4.847175955786611e-01, -4.976900851591710e-01, -4.907937552670648e-01, -4.907937552670648e-01, -5.921395128377555e-01, -2.580741676492033e-01, -2.881046374545863e-01, -3.397251448408485e-01, -3.113682886881521e-01, -3.113682886881520e-01, -4.301095339646253e-01, -5.914670307486396e-02, -8.071934983570085e-02, -3.171665753455647e-01, -1.114692780267662e-01, -1.114692780267662e-01, -1.515335668508708e-02, -1.532512858870139e-03, -3.043544926400826e-03, -1.056572212419846e-01, -4.692614875746516e-03, -4.692614875746490e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_hyb_gga_xc_b97_1p_BrOH_1_vrho():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_1p", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-2.258029971602213e+01, -2.258038102977998e+01, -2.258074517140268e+01, -2.257941356593170e+01, -2.258034258661044e+01, -2.258034258661044e+01, -3.686725769720289e+00, -3.686775387983360e+00, -3.688285842029460e+00, -3.686445739129193e+00, -3.686766659579987e+00, -3.686766659579987e+00, -7.149905961206563e-01, -7.133909115030312e-01, -6.690223831911757e-01, -6.754171728403963e-01, -7.144118620439440e-01, -7.144118620439440e-01, -2.021339185045651e-01, -2.056668233992978e-01, -8.754632838847214e-01, -1.495726923212626e-01, -2.032057582287424e-01, -2.032057582287424e-01, -1.732572598132077e-02, -1.824762237110601e-02, -7.932815057602023e-02, -7.629791996490740e-03, -1.803119612355600e-02, -1.803119612355600e-02, -5.642370920033713e+00, -5.644576231141849e+00, -5.642597166139147e+00, -5.644311987173144e+00, -5.643502270086591e+00, -5.643502270086591e+00, -1.887929948498110e+00, -1.908679267998195e+00, -1.871413567099687e+00, -1.887679460675676e+00, -1.923203621191864e+00, -1.923203621191864e+00, -6.653898134687005e-01, -7.346727076338856e-01, -6.312149342143332e-01, -6.741638665212533e-01, -6.937935748102115e-01, -6.937935748102115e-01, -1.310451890898629e-01, -1.936176496245495e-01, -1.276997029323694e-01, -2.144624057013230e+00, -1.355824118032446e-01, -1.355824118032446e-01, -7.331239043088050e-03, -8.502970122875257e-03, -6.173324147857848e-03, -9.981239314726502e-02, -7.665015312420828e-03, -7.665015312420814e-03, -6.885603590550330e-01, -6.950003397394159e-01, -6.933705238004114e-01, -6.913646103398463e-01, -6.924346066608432e-01, -6.924346066608432e-01, -6.662374242196339e-01, -5.612293824656129e-01, -6.018503514051518e-01, -6.381794505120879e-01, -6.202379413922235e-01, -6.202379413922237e-01, -7.691618726516005e-01, -2.400677845015655e-01, -2.919844313819231e-01, -3.961946942706926e-01, -3.406251842666316e-01, -3.406251842666315e-01, -5.068533210602093e-01, -7.519364984101681e-02, -9.528184121224711e-02, -3.859457532837152e-01, -1.127393522682603e-01, -1.127393522682602e-01, -2.072745909215456e-02, -2.073244287117029e-03, -4.138636405536912e-03, -1.074260452683471e-01, -6.400852892065800e-03, -6.400852892065784e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05


def test_hyb_gga_xc_b97_1p_BrOH_1_vsigma():
    # Prepare the input
    inp = test_data["BrOH"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("hyb_gga_xc_b97_1p", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vsigma"].flatten()
    ref_tgt = [-3.169689991625317e-09, -3.169632257556739e-09, -3.169391283429896e-09, -3.170336410068373e-09, -3.169659405948470e-09, -3.169659405948470e-09, -4.791576725087400e-06, -4.791253043704059e-06, -4.781572900111705e-06, -4.794229283528801e-06, -4.791329600223215e-06, -4.791329600223215e-06, -3.719550170047574e-03, -3.744460607465408e-03, -4.403186628715965e-03, -4.262492077560223e-03, -3.728587695850755e-03, -3.728587695850755e-03, -4.867534422529765e-01, -4.679972141232280e-01, -1.675418978151832e-03, -1.123484581197595e+00, -4.811247404803138e-01, -4.811247404803138e-01, -5.276434655825314e+00, -5.398363136099696e+00, -3.605941563644694e+00, -2.668081429958836e+00, -5.579670956411141e+00, -5.579670956411141e+00, -7.733201409807448e-07, -7.707941734239198e-07, -7.730643831450985e-07, -7.711002005660020e-07, -7.720199547956595e-07, -7.720199547956595e-07, -6.029613176868168e-05, -5.840196559335939e-05, -6.133365168633036e-05, -5.983037561627574e-05, -5.768210267511823e-05, -5.768210267511823e-05, -4.805145535849222e-03, -3.522406373167146e-03, -6.074355232052952e-03, -3.412196953449236e-03, -3.861542865692115e-03, -3.861542865692115e-03, -1.768801275448054e+00, -4.169944438775814e-01, -1.959793846502300e+00, -3.858421099683979e-05, -1.591627818669510e+00, -1.591627818669510e+00, -2.781827168575122e+00, -2.970199971932075e+00, -7.383779587123586e+00, -3.471565232372513e+00, -4.191048373473055e+00, -4.191048373474280e+00, -2.270933966291361e-02, -5.376182493807117e-03, -8.217902503731864e-03, -1.228710057646341e-02, -9.974199787646696e-03, -9.974199787646699e-03, -3.607444850041318e-02, -1.039645698251144e-02, -7.491149696219467e-03, -4.748680447190322e-03, -6.113449695938366e-03, -6.113449695938365e-03, -2.601144339434797e-03, -2.038415196334257e-01, -1.142797793904038e-01, -4.301776314621875e-02, -7.214360953499550e-02, -7.214360953499553e-02, -1.575100639804697e-02, -3.065005194698729e+00, -3.106865946787287e+00, -4.846440019247837e-02, -2.997146478494392e+00, -2.997146478494388e+00, -4.262429695740662e+00, -2.775501520371046e+00, -3.219873146460976e+00, -3.578329460668413e+00, -5.658047016790444e+00, -5.658047016812813e+00]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05