
import pylibxc
import pytest
import numpy
from pylibxc.example_densities import test_data


def test_lda_c_ml1_BrOH_cation_restr_1_zk():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ml1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["zk"].flatten()
    ref_tgt = [-9.893552527604409e-02, -9.893552950786262e-02, -9.893554962724059e-02, -9.893548663867703e-02, -9.893551945832943e-02, -9.893551945832943e-02, -9.199661080895813e-02, -9.199662730626737e-02, -9.199719586731456e-02, -9.199765860534982e-02, -9.199681483340147e-02, -9.199681483340147e-02, -6.855795874309852e-02, -6.853441924349389e-02, -6.795723792097855e-02, -6.813504430571216e-02, -6.809739592184437e-02, -6.809739592184437e-02, -3.670700277989649e-02, -3.704598916418379e-02, -7.174432719876363e-02, -3.075941086720105e-02, -3.319697609375819e-02, -3.319697609375819e-02, -2.172772442268852e-03, -2.282552988286120e-03, -1.101165981208172e-02, -1.275502807434908e-03, -1.593274907065998e-03, -1.593274907065998e-03, -9.461165561403423e-02, -9.461265035128498e-02, -9.461170516537118e-02, -9.461258328049364e-02, -9.461215786779184e-02, -9.461215786779184e-02, -8.643746174642940e-02, -8.652328265610430e-02, -8.637413392177179e-02, -8.645072046766246e-02, -8.651523857080262e-02, -8.651523857080262e-02, -6.563908465851662e-02, -6.746026597889876e-02, -6.389184863525312e-02, -6.476856924987186e-02, -6.599433536771274e-02, -6.599433536771274e-02, -2.379861358794352e-02, -3.674116347056021e-02, -2.246345681194350e-02, -8.620602720066273e-02, -2.671200116150321e-02, -2.671200116150321e-02, -9.896079907287037e-04, -1.247580736022829e-03, -9.593309808642374e-04, -1.649731352504223e-02, -1.151353196973470e-03, -1.151353196973470e-03, -6.574684901015988e-02, -6.561884343831377e-02, -6.566399504528891e-02, -6.570109629125366e-02, -6.568253544335929e-02, -6.568253544335929e-02, -6.513346177130210e-02, -6.164500210201856e-02, -6.271160555578134e-02, -6.371263820714508e-02, -6.320817494349264e-02, -6.320817494349264e-02, -6.847745068086300e-02, -4.215243942139012e-02, -4.654431696117429e-02, -5.332215389116544e-02, -4.991627298624498e-02, -4.991627298624497e-02, -5.970571514088246e-02, -1.060260653944437e-02, -1.379893348294003e-02, -5.242947219776711e-02, -1.996565063588243e-02, -1.996565063588242e-02, -3.022274498490130e-03, -3.398166858803358e-04, -7.075601151409677e-04, -1.901337270107872e-02, -1.067109083892829e-03, -1.067109083892828e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-08


def test_lda_c_ml1_BrOH_cation_restr_1_vrho():
    # Prepare the input
    inp = test_data["BrOH_cation_restr"]

    # Get the functional
    feval = pylibxc.LibXCFunctional("lda_c_ml1", 1)

    # Evaluate the data
    out = feval.compute(inp, do_exc=True, do_vxc=True, do_fxc=False, do_kxc=False, do_lxc=False)
    tgt = out["vrho"].flatten()
    ref_tgt = [-9.944132254977843e-02, -9.944132559967933e-02, -9.944134009965781e-02, -9.944129470410414e-02, -9.944131835706080e-02, -9.944131835706080e-02, -9.454438270885096e-02, -9.454439467581127e-02, -9.454480710236284e-02, -9.454514276469088e-02, -9.454453070572295e-02, -9.454453070572295e-02, -7.566949378619792e-02, -7.564867804502354e-02, -7.513713226481827e-02, -7.529495447439044e-02, -7.526155499608721e-02, -7.526155499608721e-02, -4.420931027855983e-02, -4.457815487244625e-02, -7.845315493784857e-02, -3.762139769317149e-02, -4.034799301828065e-02, -4.034799301828065e-02, -2.870458600405781e-03, -3.014342521747608e-03, -1.417474646294879e-02, -1.690594674497942e-03, -2.109254370245534e-03, -2.109254370245534e-03, -9.641808068975793e-02, -9.641878469919801e-02, -9.641811575908388e-02, -9.641873723111839e-02, -9.641843615266608e-02, -9.641843615266608e-02, -9.040579786809977e-02, -9.047130031459967e-02, -9.035743081206085e-02, -9.041592082744615e-02, -9.046516284422446e-02, -9.046516284422446e-02, -7.306037063869705e-02, -7.469490509008010e-02, -7.147162896979584e-02, -7.227133495149357e-02, -7.338093371692522e-02, -7.338093371692522e-02, -2.963274675866470e-02, -4.424651239677145e-02, -2.806621825948759e-02, -9.022890488130900e-02, -3.301277555337463e-02, -3.301277555337463e-02, -1.313137586303078e-03, -1.653763905998592e-03, -1.273118230277200e-03, -2.093136613358686e-02, -1.526778900849584e-03, -1.526778900849584e-03, -7.315770062048077e-02, -7.304208074094128e-02, -7.308287586817344e-02, -7.311638730823544e-02, -7.309962348822864e-02, -7.309962348822864e-02, -7.260268198679344e-02, -6.939909729005719e-02, -7.038708164329269e-02, -7.130753855245947e-02, -7.084450347450488e-02, -7.084450347450488e-02, -7.559828610229605e-02, -5.004727765129589e-02, -5.461965525325057e-02, -6.143512661508137e-02, -5.804706474221471e-02, -5.804706474221469e-02, -6.758368261458819e-02, -1.366324608198739e-02, -1.763157308833350e-02, -6.055431825482720e-02, -2.510597141472480e-02, -2.510597141472479e-02, -3.981312171773237e-03, -4.521980280577762e-04, -9.399822470510608e-04, -2.396721978546713e-02, -1.415536143563966e-03, -1.415536143563964e-03]
    error = numpy.max(numpy.abs(tgt-ref_tgt))/(1.0+numpy.max([numpy.abs(tgt), numpy.abs(ref_tgt)]))
    assert error < 5e-05